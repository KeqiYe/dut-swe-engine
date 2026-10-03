"""Bed elevation b(x,y), positive upwards, and consistent spatial derivatives."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FlatBottom:
    z: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.z): raise ValueError("bottom elevation must be finite")

    def evaluate(self, xy):
        xy = np.asarray(xy)
        return np.full(xy.shape[:-1], self.z), np.zeros_like(xy, dtype=float)

    def warp_spec(self):
        return 0, np.array([self.z,0,0,0,0,0,0,0], dtype=float), np.zeros((1,1))


@dataclass(frozen=True)
class GaussianBottom:
    height: float = .875
    center: tuple = (12.5,15.)
    sigma: float = 1.6
    base: float = 0.

    def __post_init__(self):
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be positive")
        object.__setattr__(self,"center",tuple(self.center))
        if len(self.center)!=2 or not np.isfinite([self.height,self.base,*self.center]).all():
            raise ValueError("Gaussian bottom parameters must be finite")

    def evaluate(self, xy):
        delta = np.asarray(xy)-self.center
        value = self.height*np.exp(-np.sum(delta**2,axis=-1)/(2*self.sigma**2))
        return self.base+value, -value[...,None]*delta/self.sigma**2

    def warp_spec(self):
        return 1, np.array([self.base,self.height,*self.center,self.sigma,0,0,0]), np.zeros((1,1))


@dataclass(frozen=True)
class ParabolicBottom:
    ax: float
    ay: float
    center: tuple = (0.,0.)
    base: float = 0.

    def __post_init__(self):
        object.__setattr__(self,"center",tuple(self.center))
        if len(self.center)!=2 or not np.isfinite([self.ax,self.ay,self.base,*self.center]).all():
            raise ValueError("parabolic bottom parameters must be finite")

    def evaluate(self, xy):
        delta = np.asarray(xy)-self.center
        coeff = np.array([self.ax,self.ay])
        return self.base+np.sum(coeff*delta**2,axis=-1), 2*coeff*delta

    def warp_spec(self):
        return 2, np.array([self.base,self.ax,self.ay,*self.center,0,0,0]), np.zeros((1,1))


class GridBottom:
    """Uniform raster, shape (nx,ny); bilinear value and its exact derivative.

    Bounds must cover every *current* material position, not just the initial mesh.
    Sampling a function introduces a separate terrain discretization error.
    """
    def __setattr__(self,name,value):
        if getattr(self,"_sealed",False):
            raise AttributeError("GridBottom is immutable; construct a new terrain to change it")
        object.__setattr__(self,name,value)

    def __init__(self, x, y, values):
        self.x, self.y = np.array(x,dtype=float), np.array(y,dtype=float)
        self.values = np.array(values,dtype=float,copy=True)
        for axis in (self.x,self.y):
            if axis.ndim != 1 or len(axis)<2 or not np.isfinite(axis).all():
                raise ValueError("raster axes must contain at least two finite values")
            if np.any(np.diff(axis)<=0) or not np.allclose(np.diff(axis),axis[1]-axis[0],rtol=1e-10,atol=0):
                raise ValueError("raster axes must be increasing and uniformly spaced")
        if self.values.shape != (len(self.x),len(self.y)) or not np.isfinite(self.values).all():
            raise ValueError("raster values must be finite with shape (nx,ny)")
        for array in (self.x,self.y,self.values): array.flags.writeable=False
        self._sealed=True

    @classmethod
    def from_function(cls, function, bounds, shape=(257,257)):
        """Sample function(xy)->elevations; bounds=(xmin,xmax,ymin,ymax)."""
        xmin,xmax,ymin,ymax=bounds
        x=np.linspace(xmin,xmax,shape[0]); y=np.linspace(ymin,ymax,shape[1])
        xx,yy=np.meshgrid(x,y,indexing="ij")
        xy=np.column_stack((xx.ravel(),yy.ravel()))
        values=np.broadcast_to(np.asarray(function(xy),dtype=float), (len(xy),)).reshape(shape)
        return cls(x,y,values)

    def evaluate(self, xy):
        xy=np.asarray(xy,dtype=float)
        shape=xy.shape[:-1]; p=xy.reshape(-1,2)
        dx=self.x[1]-self.x[0]; dy=self.y[1]-self.y[0]
        s=(p[:,0]-self.x[0])/dx; t=(p[:,1]-self.y[0])/dy
        if not np.isfinite(p).all() or np.any(s < -1e-10) or np.any(t < -1e-10) or np.any(s > len(self.x)-1+1e-10) or np.any(t > len(self.y)-1+1e-10):
            raise ValueError("current position is outside the bathymetry raster bounds")
        s=np.clip(s,0,len(self.x)-1); t=np.clip(t,0,len(self.y)-1)
        i=np.minimum(s.astype(int),len(self.x)-2); j=np.minimum(t.astype(int),len(self.y)-2)
        u=s-i; v=t-j
        a=self.values[i,j]; b=self.values[i+1,j]; c=self.values[i,j+1]; d=self.values[i+1,j+1]
        z=(1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*c+u*v*d
        gradient=np.column_stack((((1-v)*(b-a)+v*(d-c))/dx,((1-u)*(c-a)+u*(d-b))/dy))
        return z.reshape(shape),gradient.reshape((*shape,2))

    def warp_spec(self):
        return 3,np.array([self.x[0],self.y[0],self.x[1]-self.x[0],self.y[1]-self.y[0],0,0,0,0]),self.values
