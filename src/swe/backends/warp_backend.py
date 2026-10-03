"""Double precision, device-resident Warp implementation of the SWE kernels.

Warp is an optional dependency: importing this module requires ``warp-lang``.
``device='cpu'`` runs the identical kernels for validation without a CUDA GPU.
The public CPU fallback uses NumPy and does not require Warp.
"""

import math

import numpy as np
import warp as wp


@wp.func
def _bed_gradient(
    kind: int,
    params: wp.array(dtype=wp.float64),
    grid: wp.array2d(dtype=wp.float64),
    x: wp.float64,
    y: wp.float64,
    status: wp.array(dtype=wp.int32),
):
    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    if kind == 1:
        dx = x - params[2]
        dy = y - params[3]
        sigma2 = params[4] * params[4]
        value = params[1] * wp.exp(-(dx * dx + dy * dy) / (wp.float64(2.0) * sigma2))
        gx = -dx * value / sigma2
        gy = -dy * value / sigma2
    elif kind == 2:
        gx = wp.float64(2.0) * params[1] * (x - params[3])
        gy = wp.float64(2.0) * params[2] * (y - params[4])
    elif kind == 3:
        tx = (x - params[0]) / params[2]
        ty = (y - params[1]) / params[3]
        nx = grid.shape[0]
        ny = grid.shape[1]
        tolerance = wp.float64(1.0e-10)
        if tx < -tolerance or ty < -tolerance or tx > wp.float64(nx - 1) + tolerance or ty > wp.float64(ny - 1) + tolerance:
            wp.atomic_max(status, 0, 3)
        else:
            # Match GridBottom.evaluate at raster edges despite roundoff.
            tx = wp.clamp(tx, wp.float64(0.0), wp.float64(nx - 1))
            ty = wp.clamp(ty, wp.float64(0.0), wp.float64(ny - 1))
            ix = wp.min(int(wp.floor(tx)), nx - 2)
            iy = wp.min(int(wp.floor(ty)), ny - 2)
            fx = tx - wp.float64(ix)
            fy = ty - wp.float64(iy)
            z00 = grid[ix, iy]
            z10 = grid[ix + 1, iy]
            z01 = grid[ix, iy + 1]
            z11 = grid[ix + 1, iy + 1]
            gx = ((wp.float64(1.0) - fy) * (z10 - z00) + fy * (z11 - z01)) / params[2]
            gy = ((wp.float64(1.0) - fx) * (z01 - z00) + fx * (z11 - z10)) / params[3]
    return wp.vec2d(gx, gy)


@wp.kernel
def _force(
    points: wp.array2d(dtype=wp.float64),
    triangles: wp.array2d(dtype=wp.int32),
    grad: wp.array3d(dtype=wp.float64),
    area: wp.array(dtype=wp.float64),
    depth0: wp.array2d(dtype=wp.float64),
    q: wp.array2d(dtype=wp.float64),
    velocity: wp.array2d(dtype=wp.float64),
    gravity: wp.float64,
    density: wp.float64,
    manning: wp.float64,
    bottom_kind: int,
    bottom_params: wp.array(dtype=wp.float64),
    bottom_grid: wp.array2d(dtype=wp.float64),
    force: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    min_j: wp.float64,
):
    e = wp.tid()
    ux = wp.float64(0.0)
    uy = wp.float64(0.0)
    vx = wp.float64(0.0)
    vy = wp.float64(0.0)
    valid = True
    for a in range(3):
        node = triangles[e, a]
        u = q[node, 0]
        v = q[node, 1]
        if not wp.isfinite(u) or not wp.isfinite(v) or not wp.isfinite(velocity[node, 0]) or not wp.isfinite(velocity[node, 1]):
            valid = False
            wp.atomic_max(status, 0, 1)
        if bottom_kind == 3 and wp.isfinite(u) and wp.isfinite(v):
            # Vertices are used by snapshots/viewers and must remain within
            # the terrain even when every interior quadrature point is inside.
            sx = (points[node, 0] + u - bottom_params[0]) / bottom_params[2]
            sy = (points[node, 1] + v - bottom_params[1]) / bottom_params[3]
            tolerance = wp.float64(1.0e-10)
            if not wp.isfinite(sx) or not wp.isfinite(sy) or sx < -tolerance or sy < -tolerance or sx > wp.float64(bottom_grid.shape[0] - 1) + tolerance or sy > wp.float64(bottom_grid.shape[1] - 1) + tolerance:
                valid = False
                wp.atomic_max(status, 0, 3)
        ux = ux + u * grad[e, a, 0]
        uy = uy + u * grad[e, a, 1]
        vx = vx + v * grad[e, a, 0]
        vy = vy + v * grad[e, a, 1]
    jacobian = (wp.float64(1.0) + ux) * (wp.float64(1.0) + vy) - uy * vx
    if not wp.isfinite(jacobian) or jacobian <= min_j:
        valid = False
        wp.atomic_max(status, 0, 2)

    # Thread-local element forces: assemble only six atomic sums per element.
    local_x = wp.vec3d(wp.float64(0.0))
    local_y = wp.vec3d(wp.float64(0.0))
    if valid:
        weight = area[e] / wp.float64(3.0)
        for k in range(3):
            x = wp.float64(0.0)
            y = wp.float64(0.0)
            speed_x = wp.float64(0.0)
            speed_y = wp.float64(0.0)
            for a in range(3):
                node = triangles[e, a]
                shape = wp.float64(1.0) / wp.float64(6.0)
                if a == k:
                    shape = wp.float64(4.0) / wp.float64(6.0)
                x = x + shape * (points[node, 0] + q[node, 0])
                y = y + shape * (points[node, 1] + q[node, 1])
                speed_x = speed_x + shape * velocity[node, 0]
                speed_y = speed_y + shape * velocity[node, 1]
            if not wp.isfinite(x) or not wp.isfinite(y):
                wp.atomic_max(status, 0, 4)
            else:
                bed_grad = _bed_gradient(bottom_kind, bottom_params, bottom_grid, x, y, status)
                initial_depth = depth0[e, k]
                depth = initial_depth / jacobian
                pressure = density * gravity * initial_depth * initial_depth / (wp.float64(2.0) * jacobian * jacobian)
                bed_scale = density * gravity * initial_depth
                friction = wp.float64(0.0)
                if manning > wp.float64(0.0):
                    speed = wp.sqrt(speed_x * speed_x + speed_y * speed_y)
                    friction = bed_scale * manning * manning * speed / wp.pow(depth, wp.float64(4.0) / wp.float64(3.0))
                for a in range(3):
                    shape = wp.float64(1.0) / wp.float64(6.0)
                    if a == k:
                        shape = wp.float64(4.0) / wp.float64(6.0)
                    gx = grad[e, a, 0]
                    gy = grad[e, a, 1]
                    fx = pressure * (gx * (wp.float64(1.0) + vy) - gy * vx) - shape * (bed_scale * bed_grad[0] + friction * speed_x)
                    fy = pressure * (gy * (wp.float64(1.0) + ux) - gx * uy) - shape * (bed_scale * bed_grad[1] + friction * speed_y)
                    local_x[a] = local_x[a] + weight * fx
                    local_y[a] = local_y[a] + weight * fy
        for a in range(3):
            if not wp.isfinite(local_x[a]) or not wp.isfinite(local_y[a]):
                wp.atomic_max(status, 0, 4)
            else:
                wp.atomic_add(force, triangles[e, a], 0, local_x[a])
                wp.atomic_add(force, triangles[e, a], 1, local_y[a])


@wp.kernel
def _advance_position(
    q: wp.array2d(dtype=wp.float64),
    velocity: wp.array2d(dtype=wp.float64),
    force: wp.array2d(dtype=wp.float64),
    mass: wp.array(dtype=wp.float64),
    fixed: wp.array2d(dtype=wp.int32),
    dt: wp.float64,
    next_q: wp.array2d(dtype=wp.float64),
    predicted_v: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    i, c = wp.tid()
    value = wp.float64(0.0)
    pred = wp.float64(0.0)
    if fixed[i, c] == 0:
        value = q[i, c] + dt * (velocity[i, c] + wp.float64(0.5) * dt * force[i, c] / mass[i])
        pred = wp.float64(2.0) * (value - q[i, c]) / dt - velocity[i, c]
    if not wp.isfinite(value) or not wp.isfinite(pred):
        wp.atomic_max(status, 0, 4)
    next_q[i, c] = value
    predicted_v[i, c] = pred


@wp.kernel
def _advance_velocity(
    q: wp.array2d(dtype=wp.float64),
    next_q: wp.array2d(dtype=wp.float64),
    force: wp.array2d(dtype=wp.float64),
    mass: wp.array(dtype=wp.float64),
    fixed: wp.array2d(dtype=wp.int32),
    dt: wp.float64,
    next_v: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
):
    i, c = wp.tid()
    value = wp.float64(0.0)
    if fixed[i, c] == 0:
        value = (next_q[i, c] - q[i, c]) / dt + wp.float64(0.5) * dt * force[i, c] / mass[i]
    if not wp.isfinite(value):
        wp.atomic_max(status, 0, 4)
    next_v[i, c] = value


def available():
    """Whether Warp can initialize at least one CUDA device."""
    try:
        return bool(wp.get_cuda_devices())
    except Exception:
        return False


class WarpBackend:
    """Preallocated explicit Verlet solver using Warp CPU or CUDA kernels.

    A backend owns scratch arrays and is not reentrant. Each simulation should
    own its own backend. All array arguments must reside on ``self.device``.
    Failed steps leave both output arrays unchanged, including in-place calls.
    """

    name = "warp"

    def __init__(self, model, device="cuda:0"):
        try:
            self.device = wp.get_device(device)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Requested Warp device {device!r} is unavailable: {exc}") from exc
        self.model = model
        self.node_count = len(model.points)
        self.element_count = len(model.triangles)
        self.points = self.array(model.points)
        self.triangles = wp.array(np.asarray(model.triangles, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.grad = self.array(model.grad)
        self.area = self.array(model.area)
        self.depth0 = self.array(model.depth0)
        self.mass = self.array(model.mass)
        self.fixed = wp.array(np.asarray(model.fixed, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.gravity = float(model.gravity)
        self.density = float(model.density)
        self.manning = float(model.manning)
        kind, params, grid = model.bottom.warp_spec()
        self.bottom_kind = int(kind)
        params = np.asarray(params, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)
        if self.bottom_kind not in (0, 1, 2, 3):
            raise ValueError(f"Unsupported CUDA bathymetry kind: {kind}")
        if params.shape != (8,) or not np.isfinite(params).all():
            raise ValueError("Bathymetry warp_spec parameters must be eight finite float64 values")
        if self.bottom_kind == 1 and params[4] <= 0:
            raise ValueError("Gaussian bathymetry sigma must be positive")
        if self.bottom_kind == 3:
            if grid.ndim != 2 or min(grid.shape) < 2 or not np.isfinite(grid).all():
                raise ValueError("Sampled bathymetry needs a finite grid of at least 2 by 2 points")
            if params[2] <= 0 or params[3] <= 0:
                raise ValueError("Sampled bathymetry grid spacing must be positive")
        else:
            # Non-grid functions still pass a typed valid array to the kernel.
            grid = np.zeros((2, 2), dtype=np.float64)
        self.bottom_params = self.array(params)
        self.bottom_grid = self.array(grid)
        shape = (self.node_count, 2)
        self._q_next = self.zeros(shape)
        self._v_next = self.zeros(shape)
        self._v_predicted = self.zeros(shape)
        self._force0 = self.zeros(shape)
        self._force1 = self.zeros(shape)
        self._status = wp.zeros(1, dtype=wp.int32, device=self.device)

    def array(self, values):
        """Copy a host array into a float64 array on the simulation device."""
        return wp.array(np.ascontiguousarray(values, dtype=np.float64), dtype=wp.float64, device=self.device)

    def zeros(self, shape):
        return wp.zeros(shape=shape, dtype=wp.float64, device=self.device)

    @staticmethod
    def numpy(values):
        """Return an independent host copy, synchronizing the producing work."""
        return values.numpy().copy()

    def copy(self, dst, src):
        wp.copy(dst, src)

    def step(self, q, v, out_q, out_v, dt, min_j=1e-8):
        """Advance one time step; reject invalid candidates transactionally."""
        dt = float(dt)
        min_j = float(min_j)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(min_j) or min_j <= 0:
            raise ValueError("min_j must be finite and positive")
        for name, array in (("q", q), ("v", v), ("out_q", out_q), ("out_v", out_v)):
            if array.shape != (self.node_count, 2) or array.dtype != wp.float64:
                raise ValueError(f"{name} must be a float64 Warp array of shape ({self.node_count}, 2)")
            if array.device != self.device:
                raise ValueError(f"{name} must reside on {self.device}")
        if out_q.ptr == out_v.ptr:
            raise ValueError("Displacement and velocity outputs must be distinct arrays")
        self._status.zero_()
        common = [self.points, self.triangles, self.grad, self.area, self.depth0]
        physical = [self.gravity, self.density, self.manning, self.bottom_kind,
                    self.bottom_params, self.bottom_grid]
        self._force0.zero_()
        wp.launch(_force, dim=self.element_count,
                  inputs=common + [q, v] + physical + [self._force0, self._status, min_j],
                  device=self.device)
        wp.launch(_advance_position, dim=(self.node_count, 2),
                  inputs=[q, v, self._force0, self.mass, self.fixed, dt,
                          self._q_next, self._v_predicted, self._status], device=self.device)
        self._force1.zero_()
        wp.launch(_force, dim=self.element_count,
                  inputs=common + [self._q_next, self._v_predicted] + physical + [self._force1, self._status, min_j],
                  device=self.device)
        wp.launch(_advance_velocity, dim=(self.node_count, 2),
                  inputs=[q, self._q_next, self._force1, self.mass, self.fixed, dt,
                          self._v_next, self._status], device=self.device)
        # Only this one integer is transferred to the host during a time step.
        error = int(self._status.numpy()[0])
        if error:
            messages = {
                1: "Non-finite displacement or velocity in input state",
                2: f"Invalid or inverted mesh: element Jacobian is non-finite or <= {min_j:g}",
                3: "A moved quadrature point lies outside the sampled bathymetry domain",
                4: "Non-finite force, displacement, or velocity during the time step",
            }
            raise RuntimeError(messages.get(error, "Unknown Warp solver failure"))
        wp.copy(out_q, self._q_next)
        wp.copy(out_v, self._v_next)
