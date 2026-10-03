"""Public model/state/solver API for the nondispersive displacement equations."""
from pathlib import Path
import hashlib
import copy
import numpy as np
from .mesh import Mesh
from .terrain import FlatBottom, GridBottom
from .backends.numpy_backend import NumpyBackend, QUADRATURE, geometry


class NumericalError(RuntimeError):
    """A candidate step was rejected; accepted input state remains unchanged."""


def _field(value, xy, count, components=None):
    a=np.asarray(value(xy) if callable(value) else value,dtype=np.float64)
    shape=(count,) if components is None else (count,components)
    try:
        a=np.array(np.broadcast_to(a,shape),copy=True)
    except ValueError as e:
        raise ValueError(f"field must broadcast to {shape}, received {a.shape}") from e
    if not np.isfinite(a).all():
        raise ValueError("initial fields must be finite")
    return a


class ModelBuilder:
    def __init__(self,gravity=9.81,density=1000.,manning=0.):
        for name,value in (("gravity",gravity),("density",density)):
            if not np.isfinite(value) or value<=0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(manning) or manning<0:
            raise ValueError("manning must be finite and nonnegative")
        self.gravity=float(gravity); self.density=float(density); self.manning=float(manning)
        self._water=None

    def add_water(self,*,mesh,bathymetry=None,surface_height=1.,velocity=(0.,0.),boundaries=None):
        """Add one connected mesh. boundary mask True constrains that component.

        surface_height is absolute elevation: scalar, nodal array or function(xy).
        velocity is a constant pair, nodal (N,2) array or function(xy).
        For an arbitrary bed function use GridBottom.from_function explicitly.
        """
        if self._water is not None:
            raise ValueError("first release supports one water mesh per model")
        if not isinstance(mesh,Mesh):
            raise TypeError("mesh must be a swe.Mesh")
        if boundaries is not None:
            mesh=Mesh(mesh.points,mesh.triangles,boundaries)
        bottom=FlatBottom() if bathymetry is None else bathymetry
        if not hasattr(bottom,"evaluate") or not hasattr(bottom,"warp_spec"):
            raise TypeError("bathymetry needs evaluate/warp_spec; sample functions using GridBottom.from_function")
        self._water=(mesh,bottom,surface_height,velocity)
        return self

    def finalize(self,device="auto",dtype="float64"):
        if dtype not in ("float64",np.float64):
            raise ValueError("first release validates float64 only")
        if self._water is None:
            raise ValueError("add_water must be called before finalize")
        mesh,bottom,eta,velocity=self._water
        return Model(mesh,bottom,eta,velocity,self.gravity,self.density,self.manning,device)


class Model:
    def __setattr__(self,name,value):
        if getattr(self,"_sealed",False):
            raise AttributeError("finalized models are immutable; use a new ModelBuilder to change parameters")
        object.__setattr__(self,name,value)

    def __init__(self,mesh,bottom,eta,velocity,gravity,density,manning,device):
        # A builder may outlive edits to the caller's mutable Mesh instance.
        mesh=Mesh(mesh.points,mesh.triangles,mesh.fixed)
        self.points=mesh.points.copy(); self.triangles=mesh.triangles.copy(); self.fixed=mesh.fixed.copy()
        bottom=(GridBottom(bottom.x,bottom.y,bottom.values) if isinstance(bottom,GridBottom) else copy.deepcopy(bottom))
        self.bottom=bottom; self.gravity=gravity; self.density=density; self.manning=manning
        p=self.points[self.triangles]; x=p[:,:,0]; y=p[:,:,1]
        twice=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(y[:,1]-y[:,0])*(x[:,2]-x[:,0])
        self.area=twice/2
        gx=np.column_stack((y[:,1]-y[:,2],y[:,2]-y[:,0],y[:,0]-y[:,1]))/twice[:,None]
        gy=np.column_stack((x[:,2]-x[:,1],x[:,0]-x[:,2],x[:,1]-x[:,0]))/twice[:,None]
        self.grad=np.stack((gx,gy),axis=-1)
        self.qref=np.einsum('qi,eia->eqa',QUADRATURE,p)
        self.surface0=_field(eta,self.points,len(self.points))
        if callable(eta):
            etaq=_field(eta,self.qref.reshape(-1,2),len(self.triangles)*3).reshape(-1,3)
        else:
            etaq=np.einsum('qi,ei->eq',QUADRATURE,self.surface0[self.triangles])
        bq,bg=bottom.evaluate(self.qref); bn,bgn=bottom.evaluate(self.points)
        if not all(np.isfinite(a).all() for a in (bq,bg,bn,bgn)):
            raise ValueError("bathymetry values and gradients must be finite")
        self.depth0=etaq-bq; self.node_depth0=self.surface0-bn
        if np.any(self.depth0<=0) or np.any(self.node_depth0 < -1e-10):
            raise ValueError("mesh must cover the initially wet region; quadrature depth must be positive")
        localmass=self.density*self.area[:,None]/3*np.einsum('eq,qi->ei',self.depth0,QUADRATURE)
        self.mass=np.bincount(self.triangles.ravel(),weights=localmass.ravel(),minlength=len(self.points))
        if not np.isfinite(self.mass).all() or np.any(self.mass<=0):
            raise ValueError("all lumped nodal masses must be finite and positive")
        self.initial_velocity=_field(velocity,self.points,len(self.points),2)
        self.initial_velocity[self.fixed]=0
        self.volume0=float(np.sum(self.area*self.depth0.mean(axis=1)))
        self.backend_reason="explicit NumPy CPU backend"
        if device not in ("auto","cpu","cuda","cuda:0","warp:cpu") and not str(device).startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda[:index], or warp:cpu")
        if device=="cpu":
            self.backend=NumpyBackend(self)
        else:
            try:
                from .backends.warp_backend import WarpBackend, available
            except ImportError as e:
                if device!="auto":
                    raise RuntimeError("install dut-swe-engine[cuda] to use Warp/CUDA") from e
                self.backend=NumpyBackend(self)
                self.backend_reason="CUDA optional dependency not installed; using CPU"
            else:
                if device=="auto" and not available():
                    self.backend=NumpyBackend(self)
                    self.backend_reason="CUDA unavailable; using CPU"
                else:
                    target="cpu" if device=="warp:cpu" else ("cuda:0" if device in ("auto","cuda") else device)
                    self.backend=WarpBackend(self,device=target)
                    self.backend_reason="Warp native kernels"
        self.device=str(getattr(self.backend,"device",device))
        # Fingerprint initial model, including bed interpolation and constraints.
        digest=hashlib.sha256()
        kind,params,grid=self.bottom.warp_spec()
        for a in (self.points,self.triangles,self.fixed,self.depth0,self.node_depth0,self.initial_velocity,
                  np.array([gravity,density,manning,kind]),params,grid):
            a=np.ascontiguousarray(a)
            digest.update(str((a.shape,a.dtype.str)).encode()); digest.update(a.tobytes())
        self.fingerprint=digest.hexdigest()
        for a in (self.points,self.triangles,self.fixed,self.grad,self.area,self.depth0,self.mass,
                  self.qref,self.surface0,self.node_depth0,self.initial_velocity):
            a.flags.writeable=False
        self._sealed=True

    def state(self):
        return State(self)

    def load_state(self,path):
        state=self.state()
        with np.load(path,allow_pickle=False) as data:
            if str(data["model_fingerprint"])!=self.fingerprint:
                raise ValueError("checkpoint belongs to a different model")
            q=np.array(data["displacement"]); v=np.array(data["velocity"])
            raw_time=data["time"]; raw_steps=data["step_index"]
            if raw_time.shape!=() or raw_steps.shape!=() or raw_steps.dtype.kind not in "iu":
                raise ValueError("checkpoint time and step_index must be scalar, with integer step_index")
            time=float(raw_time); steps=int(raw_steps)
        if q.shape!=self.points.shape or v.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("invalid checkpoint state arrays")
        if not np.isfinite(time) or time<0 or steps<0 or np.any(q[self.fixed]!=0) or np.any(v[self.fixed]!=0):
            raise ValueError("invalid checkpoint time or wall constraints")
        self.backend.copy(state.displacement,self.backend.array(q)); self.backend.copy(state.velocity,self.backend.array(v))
        state.time=time; state.step_index=steps
        self.diagnostics(state)
        return state

    def diagnostics(self,state):
        self._check_state(state)
        q=self.backend.numpy(state.displacement); v=self.backend.numpy(state.velocity)
        jac,current,*_=geometry(self,q)
        if not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite(jac).all() or np.any(jac<=0):
            raise NumericalError("invalid state: nonfinite value or inverted element")
        bed,_=self.bottom.evaluate(current)
        depth=self.depth0/jac[:,None]
        pe=np.sum(self.area[:,None]/3*self.density*self.gravity*(self.depth0**2/(2*jac[:,None])+self.depth0*bed))
        ke=.5*np.sum(self.mass[:,None]*v*v)
        volume=float(np.sum(self.area*jac*depth.mean(axis=1)))
        return dict(time=float(state.time),energy=float(pe+ke),potential_energy=float(pe),kinetic_energy=float(ke),
                    volume=volume,relative_volume_error=abs(volume-self.volume0)/self.volume0,
                    min_j=float(jac.min()),max_j=float(jac.max()),min_depth=float(depth.min()),
                    max_speed=float(np.linalg.norm(v,axis=1).max()))

    def _check_state(self,state):
        if not isinstance(state,State) or state.model is not self:
            raise ValueError("state belongs to a different model")

    def suggested_dt(self,state=None,cfl=.15):
        """Conservative geometric estimate, not a guarantee against mesh tangling."""
        if not np.isfinite(cfl) or cfl<=0:
            raise ValueError("cfl must be positive")
        q=np.zeros_like(self.points) if state is None else self.backend.numpy(state.displacement)
        p=(self.points+q)[self.triangles]
        edge=np.max(np.linalg.norm(p-np.roll(p,1,axis=1),axis=-1),axis=1)
        jac,*_=geometry(self,q)
        if np.any(jac<=0): raise NumericalError("inverted mesh")
        altitude=2*self.area*jac/edge
        speed=0 if state is None else self.diagnostics(state)["max_speed"]
        wave=np.sqrt(self.gravity*np.max(self.depth0/jac[:,None],axis=1))
        return float(cfl*np.min(altitude/(wave+speed)))


class State:
    def __init__(self,model):
        self.model=model
        self.displacement=model.backend.zeros(model.points.shape)
        self.velocity=model.backend.array(model.initial_velocity)
        self.time=0.; self.step_index=0

    def reset(self):
        self.model.backend.copy(self.displacement,self.model.backend.zeros(self.model.points.shape))
        self.model.backend.copy(self.velocity,self.model.backend.array(self.model.initial_velocity))
        self.time=0.; self.step_index=0

    def to_numpy(self):
        m=self.model; q=m.backend.numpy(self.displacement); v=m.backend.numpy(self.velocity)
        jac,*_=geometry(m,q)
        if not np.isfinite(jac).all() or np.any(jac<=0):
            raise NumericalError("cannot reconstruct an invalid mesh")
        # Historical display reconstruction: average incident element estimates.
        nodaldepth=m.node_depth0[m.triangles]/jac[:,None]
        count=np.bincount(m.triangles.ravel(),minlength=len(m.points))
        depth=np.bincount(m.triangles.ravel(),weights=nodaldepth.ravel(),minlength=len(m.points))/count
        xy=m.points+q; bed,_=m.bottom.evaluate(xy); surface=bed+depth
        return dict(displacement=q,velocity=v,points=np.column_stack((xy,surface)),surface_height=surface,
                    depth=depth,bed=bed,element_jacobian=jac,time=float(self.time),step_index=self.step_index)

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,model_fingerprint=self.model.fingerprint,
                            displacement=self.model.backend.numpy(self.displacement),
                            velocity=self.model.backend.numpy(self.velocity),time=self.time,step_index=self.step_index)


class DisplacementSolver:
    def __init__(self,model,min_j=1e-8):
        if not np.isfinite(min_j) or min_j<=0: raise ValueError("min_j must be positive")
        self.model=model; self.min_j=min_j

    def step(self,state_in,state_out,dt):
        """Advance one fixed step. Output is committed only after validation."""
        if not np.isfinite(dt) or dt<=0: raise ValueError("dt must be finite and positive")
        self.model._check_state(state_in); self.model._check_state(state_out)
        if state_in is state_out: raise ValueError("use distinct input/output states and swap them after step")
        inputs=(state_in.displacement,state_in.velocity); outputs=(state_out.displacement,state_out.velocity)
        for source in inputs:
            for target in outputs:
                aliased=(np.shares_memory(source,target) if isinstance(source,np.ndarray) and isinstance(target,np.ndarray)
                         else getattr(source,"ptr",None)==getattr(target,"ptr",-1))
                if aliased: raise ValueError("input and output state arrays must not alias")
        next_time=state_in.time+dt
        if not np.isfinite(state_in.time) or state_in.time<0 or not np.isfinite(next_time) or next_time<=state_in.time:
            raise ValueError("dt must advance a finite simulation time; increase dt or reset the time origin")
        if not isinstance(state_in.step_index,(int,np.integer)) or state_in.step_index<0:
            raise ValueError("step_index must be a nonnegative integer")
        try:
            self.model.backend.step(state_in.displacement,state_in.velocity,state_out.displacement,state_out.velocity,float(dt),self.min_j)
        except (RuntimeError,ValueError) as e:
            # Classify known physical validity errors only. Device failures propagate.
            if any(word in str(e).lower() for word in ("jacobian","nonfinite","non-finite","bathymetry","inverted","terrain")):
                raise NumericalError(str(e)) from e
            raise
        state_out.time=next_time; state_out.step_index=state_in.step_index+1

    def advance(self,state,duration,max_dt,adaptive=False,max_subdivisions=12):
        """Advance in place through accepted substeps; return accepted step count.

        On failure, `state` retains the last accepted substep, not the start of
        the entire requested duration. Adaptive mode halves rejected steps.
        """
        if not np.isfinite(duration) or duration<0 or not np.isfinite(max_dt) or max_dt<=0:
            raise ValueError("duration must be nonnegative and max_dt positive")
        if not isinstance(max_subdivisions,int) or max_subdivisions<0:
            raise ValueError("max_subdivisions must be a nonnegative integer")
        self.model._check_state(state)
        end=state.time+duration; scratch=self.model.state(); count=0
        if not np.isfinite(end) or not np.isfinite(state.time) or state.time<0 or (duration>0 and end<=state.time):
            raise ValueError("duration must advance a finite simulation time")
        while state.time<end:
            dt=min(max_dt,end-state.time)
            for retry in range(max_subdivisions+1):
                try:
                    self.step(state,scratch,dt)
                    break
                except NumericalError:
                    if not adaptive or retry==max_subdivisions: raise
                    dt*=.5
            self.model.backend.copy(state.displacement,scratch.displacement)
            self.model.backend.copy(state.velocity,scratch.velocity)
            state.time=scratch.time; state.step_index=scratch.step_index; count+=1
        return count
