"""Vectorized float64 reference backend, matching the historical MATLAB kernel."""
import numpy as np
from ..terrain import GridBottom

QUADRATURE = np.array([[4.,1.,1.],[1.,4.,1.],[1.,1.,4.]])/6.


def geometry(model, q):
    qe=q[model.triangles]
    # F[a,b] = delta_ab + derivative of displacement component a in direction b.
    gradient=np.einsum('eia,eib->eab',qe,model.grad)
    ux=gradient[:,0,0]; uy=gradient[:,0,1]
    vx=gradient[:,1,0]; vy=gradient[:,1,1]
    jac=(1+ux)*(1+vy)-uy*vx
    current=np.einsum('qi,eia->eqa',QUADRATURE,model.points[model.triangles]+qe)
    return jac,current,ux,uy,vx,vy


class NumpyBackend:
    device="cpu"

    def __init__(self,model):
        self.model=model

    @staticmethod
    def array(a): return np.array(a,dtype=np.float64,copy=True)
    @staticmethod
    def zeros(shape): return np.zeros(shape,dtype=np.float64)
    @staticmethod
    def numpy(a): return np.array(a,copy=True)
    @staticmethod
    def copy(dst,src): np.copyto(dst,src)

    def force(self,q,v,min_j=1e-8):
        m=self.model
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise RuntimeError("nonfinite displacement or velocity")
        if isinstance(m.bottom,GridBottom):
            m.bottom.evaluate(m.points+q)
        jac,current,ux,uy,vx,vy=geometry(m,q)
        if not np.isfinite(jac).all() or np.any(jac <= min_j):
            raise RuntimeError(f"element Jacobian below safety threshold: min(J)={np.min(jac):.6g}")
        _,bedgrad=m.bottom.evaluate(current)
        if not np.isfinite(bedgrad).all():
            raise RuntimeError("nonfinite bathymetry gradient")
        gx=m.grad[:,:,0]; gy=m.grad[:,:,1]
        cofactor=np.stack((gx*(1+vy[:,None])-gy*vx[:,None],gy*(1+ux[:,None])-gx*uy[:,None]),axis=-1)
        pressure=m.density*m.gravity*m.depth0**2/(2*jac[:,None]**2)
        # Each element owns three local force vectors; scatter to shared nodes.
        fe=cofactor*pressure.sum(axis=1)[:,None,None]
        fe-=m.density*m.gravity*np.einsum('eq,eqa,qi->eia',m.depth0,bedgrad,QUADRATURE)
        if m.manning:
            vq=np.einsum('qi,eia->eqa',QUADRATURE,v[m.triangles])
            depth=m.depth0/jac[:,None]
            drag=m.density*m.gravity*m.manning**2*m.depth0*np.linalg.norm(vq,axis=-1)/depth**(4/3)
            fe-=np.einsum('eq,eqa,qi->eia',drag,vq,QUADRATURE)
        fe*=m.area[:,None,None]/3
        force=np.column_stack([np.bincount(m.triangles.ravel(),weights=fe[:,:,c].ravel(),minlength=len(m.points)) for c in range(2)])
        force[m.fixed]=0
        if not np.isfinite(force).all():
            raise RuntimeError("nonfinite assembled force")
        return force

    def step(self,q,v,out_q,out_v,dt,min_j=1e-8):
        m=self.model
        f0=self.force(q,v,min_j)
        q1=q+dt*(v+.5*dt*f0/m.mass[:,None])
        q1[m.fixed]=0
        predicted=2*(q1-q)/dt-v
        f1=self.force(q1,predicted,min_j)
        v1=(q1-q)/dt+.5*dt*f1/m.mass[:,None]
        v1[m.fixed]=0
        if not np.isfinite(v1).all():
            raise RuntimeError("nonfinite candidate velocity")
        # Commit only after both force evaluations and all validity checks pass.
        np.copyto(out_q,q1); np.copyto(out_v,v1)
