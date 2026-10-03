"""Reference horizontal triangle meshes; coordinates are in metres."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    points: np.ndarray
    triangles: np.ndarray
    fixed: np.ndarray | None = None

    def __post_init__(self):
        self.points = np.array(self.points, dtype=np.float64, copy=True)
        raw = np.asarray(self.triangles)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("triangles must be an integer (E,3) array")
        self.triangles = np.array(raw, dtype=np.int32, copy=True)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or not np.isfinite(self.points).all():
            raise ValueError("points must be finite horizontal coordinates of shape (N,2)")
        if len(raw) == 0 or raw.min() < 0 or raw.max() >= len(self.points):
            raise ValueError("triangle vertex indices are empty or out of range")
        p = self.points[self.triangles]
        a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        cross = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
        if np.any(cross <= 0):
            raise ValueError("triangles must be nondegenerate and counterclockwise")
        if len(np.unique(raw)) != len(self.points):
            raise ValueError("mesh contains unused nodes")
        self.fixed = (np.zeros_like(self.points, dtype=bool) if self.fixed is None
                      else np.array(self.fixed, dtype=bool, copy=True))
        if self.fixed.shape != self.points.shape:
            raise ValueError("fixed must be a boolean (N,2) mask of constrained components")

    @classmethod
    def rectangle(cls, nx, ny, size=(10., 2.), origin=(0., 0.), walls=True):
        if not isinstance(nx, int) or not isinstance(ny, int) or nx < 1 or ny < 1:
            raise ValueError("nx and ny must be positive integer cell counts")
        if len(size) != 2 or not np.isfinite(size).all() or min(size) <= 0:
            raise ValueError("rectangle size must be two positive lengths")
        x = np.linspace(origin[0], origin[0]+size[0], nx+1)
        y = np.linspace(origin[1], origin[1]+size[1], ny+1)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel()))
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        a = (i*(ny+1)+j).ravel(); b = a+ny+1
        triangles = np.concatenate((np.column_stack((a,b,b+1)), np.column_stack((a,b+1,a+1))))
        fixed = np.zeros_like(points, dtype=bool)
        if walls:
            fixed[:,0] = (xx.ravel() == x[0]) | (xx.ravel() == x[-1])
            fixed[:,1] = (yy.ravel() == y[0]) | (yy.ravel() == y[-1])
        return cls(points, triangles, fixed)

    def refined(self, levels=1):
        if not isinstance(levels, int) or levels < 0:
            raise ValueError("levels must be a nonnegative integer")
        mesh = self
        for _ in range(levels):
            t = mesh.triangles
            edges = np.sort(np.concatenate((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]])), axis=1)
            unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
            mid = len(mesh.points)+inverse.reshape(3,-1).T
            points = np.concatenate((mesh.points, mesh.points[unique].mean(axis=1)))
            fixed = np.concatenate((mesh.fixed, mesh.fixed[unique].all(axis=1) & (counts==1)[:,None]))
            a,b,c=t.T; ab,bc,ca=mid.T
            triangles = np.concatenate((np.column_stack((a,ab,ca)),np.column_stack((ab,b,bc)),
                                        np.column_stack((ca,bc,c)),np.column_stack((ab,bc,ca))))
            mesh = Mesh(points, triangles, fixed)
        return mesh
