"""Small, reproducible benchmark scenes.

Every factory returns ``(model, metadata)``. Analytic fields, where supplied,
are continuous references, not values produced by the numerical solver.
"""

from importlib.resources import files

import numpy as np


def _positive(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def island(device="auto", refinements=0, gravity=9.81):
    """The 547-node base island case, optionally refined.

    ``refinements=3`` gives 64,384 triangles (64 times the base mesh).
    Refinement projects the island shoreline onto the initial zero-depth
    contour. Merely bisecting straight coastal edges would put their new
    vertices on the dry side of the curved island shore.
    """
    from . import GaussianBottom, Mesh, ModelBuilder

    gravity = _positive("gravity", gravity)
    with files("swe").joinpath("data/island_mesh.npz").open("rb") as stream:
        with np.load(stream) as data:
            mesh = Mesh(data["points"], data["triangles"], data["fixed"])
    if isinstance(refinements, bool) or not isinstance(refinements, int) or refinements < 0:
        raise ValueError("refinements must be a nonnegative integer")
    mean_height, amplitude = 0.32, 0.032
    k = np.sqrt(3 * amplitude / (4 * mean_height**2 * (mean_height + amplitude)))
    wave_center = 5 / k
    c = np.sqrt(gravity * (mean_height + amplitude))

    def profile(xy):
        return 1 / np.cosh(k * (np.asarray(xy)[:, 0] - wave_center)) ** 2

    def surface(xy):
        return mean_height + amplitude * profile(xy)

    bottom = GaussianBottom(height=0.875, center=(12.5, 15.0), sigma=1.6)
    for _ in range(refinements):
        mesh = mesh.refined(1)
        # Locate free shoreline nodes from edge incidence. Outer walls have
        # at least one constrained component, while the island edge is free.
        triangles = mesh.triangles
        edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
        unique, count = np.unique(edges, axis=0, return_counts=True)
        boundary = np.unique(unique[count == 1])
        shoreline = boundary[~mesh.fixed[boundary].any(axis=1)]
        center = np.asarray(bottom.center)
        rays = mesh.points[shoreline] - center
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        low = np.zeros(len(shoreline))
        high = np.full(len(shoreline), 8 * bottom.sigma)
        for _ in range(60):
            middle = (low + high) / 2
            xy = center + middle[:, None] * rays
            wet = surface(xy) >= bottom.evaluate(xy)[0]
            high = np.where(wet, middle, high)
            low = np.where(wet, low, middle)
        points = mesh.points.copy()
        # Choose the wet-side bracket, preserving physical zero depth without
        # imposing an artificial minimum depth or modifying bathymetry.
        points[shoreline] = center + high[:, None] * rays
        mesh = Mesh(points, mesh.triangles, mesh.fixed)

    def velocity(xy):
        value = np.zeros_like(xy, dtype=float)
        value[:, 0] = c * k * np.sqrt(4 * mean_height * amplitude / 3) * profile(xy)
        return value

    builder = ModelBuilder(gravity=gravity, density=1000.0)
    builder.add_water(
        mesh=mesh,
        bathymetry=bottom,
        surface_height=surface,
        velocity=velocity,
    )
    return builder.finalize(device=device), {
        "name": "island",
        "dt": 0.00625 / 2**refinements * np.sqrt(9.81 / gravity),
        "duration": (10.0 if refinements==0 else 3.0) * np.sqrt(9.81 / gravity),
        "validation_limit": ("Base mesh verified for 10 s" if refinements==0 else
                             "Refined island verified for 3 s only; compression causes rejection near 4 s at level 3"),
        "description": "Gaussian island and sech-squared pulse on the base mesh; nondispersive model.",
        "reference": "Frozen 10 s baseline on the base mesh at dt=0.00625 and gravity=9.81",
        "refinements": refinements,
    }


def standing_wave(
    device="auto", nx=40, ny=4, length=10.0, width=1.0,
    depth=1.0, amplitude=0.001, gravity=9.81,
):
    """Small-amplitude first longitudinal standing mode in a closed tank.

    ``metadata['analytic'](xy, time)`` returns Eulerian ``surface_height`` and
    ``velocity`` fields. This reference solves the *linearized* shallow-water
    equations. The numerical model is nonlinear, so finite-amplitude error
    remains even as the mesh and time step tend to zero.
    """
    from . import FlatBottom, Mesh, ModelBuilder

    length = _positive("length", length)
    width = _positive("width", width)
    depth = _positive("depth", depth)
    gravity = _positive("gravity", gravity)
    amplitude = float(amplitude)
    if not np.isfinite(amplitude) or abs(amplitude) >= depth:
        raise ValueError("amplitude must be finite and have magnitude smaller than depth")
    mesh = Mesh.rectangle(nx=nx, ny=ny, size=(length, width), walls=True)
    k = np.pi / length
    omega = k * np.sqrt(gravity * depth)

    def analytic(xy, time):
        xy = np.asarray(xy, dtype=float)
        velocity = np.zeros_like(xy)
        velocity[:, 0] = amplitude * gravity * k / omega * np.sin(k * xy[:, 0]) * np.sin(omega * time)
        return {
            "surface_height": depth + amplitude * np.cos(k * xy[:, 0]) * np.cos(omega * time),
            "velocity": velocity,
        }

    builder = ModelBuilder(gravity=gravity, density=1000.0)
    builder.add_water(
        mesh=mesh, bathymetry=FlatBottom(z=0.0),
        surface_height=lambda xy: analytic(xy, 0.0)["surface_height"],
        velocity=(0.0, 0.0),
    )
    return builder.finalize(device=device), {
        "name": "standing_wave",
        "dt": 0.1 * min(length / nx, width / ny) / np.sqrt(gravity * (depth + abs(amplitude))),
        "duration": 2 * np.pi / omega,
        "period": 2 * np.pi / omega,
        "omega": omega,
        "amplitude": amplitude,
        "description": "Small-amplitude flat-bed standing wave with impermeable walls.",
        "analytic": analytic,
        "reference_kind": "linearized, Eulerian; nonlinear truncation error is expected",
    }


def _disk_mesh(radius, center, rings, sectors):
    from . import Mesh

    if isinstance(rings, bool) or int(rings) != rings or rings < 1:
        raise ValueError("rings must be an integer >= 1")
    if isinstance(sectors, bool) or int(sectors) != sectors or sectors < 8:
        raise ValueError("sectors must be an integer >= 8")
    rings, sectors = int(rings), int(sectors)
    angles = 2 * np.pi * np.arange(sectors) / sectors
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    points = [np.asarray(center, dtype=float)]
    for ring in range(1, rings + 1):
        points.extend(np.asarray(center) + radius * ring / rings * circle)
    triangles = []
    for sector in range(sectors):
        triangles.append((0, 1 + sector, 1 + (sector + 1) % sectors))
    for ring in range(1, rings):
        inner = 1 + (ring - 1) * sectors
        outer = 1 + ring * sectors
        for sector in range(sectors):
            following = (sector + 1) % sectors
            triangles.extend([
                (inner + sector, outer + sector, outer + following),
                (inner + sector, outer + following, inner + following),
            ])
    return Mesh(np.asarray(points), np.asarray(triangles, dtype=np.int32))


def sloshing(
    device="auto", rings=8, sectors=48, radius=10.0,
    depth=1.0, amplitude=0.5, gravity=9.81,
):
    """Planar free-surface translation in a circular parabolic basin.

    With ``a=depth/radius**2`` and ``omega=sqrt(2*g*a)``, the wet disk center
    is ``q(t)=(amplitude*cos(omega*t), 0)``. This is an exact nonlinear
    shallow-water solution with a translating shoreline. The supplied mesh
    approximates that shoreline by straight polygon edges.

    ``metadata['analytic'](reference_xy, time)`` accepts *initial material
    coordinates*, unlike the Eulerian standing-wave reference, and returns
    displacement, velocity, current points, depth and surface height.
    This case does not establish general dry-land inundation support.
    """
    from . import ModelBuilder, ParabolicBottom

    radius = _positive("radius", radius)
    depth = _positive("depth", depth)
    gravity = _positive("gravity", gravity)
    amplitude = float(amplitude)
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    a = depth / radius**2
    omega = np.sqrt(2 * gravity * a)
    center = np.array([amplitude, 0.0])
    mesh = _disk_mesh(radius, center, rings, sectors)

    def surface(xy):
        return depth + 2 * a * amplitude * xy[:, 0] - a * amplitude**2

    def analytic(reference_xy, time):
        reference_xy = np.asarray(reference_xy, dtype=float)
        displacement = np.zeros_like(reference_xy)
        displacement[:, 0] = amplitude * (np.cos(omega * time) - 1)
        velocity = np.zeros_like(reference_xy)
        velocity[:, 0] = -amplitude * omega * np.sin(omega * time)
        xy = reference_xy + displacement
        water_depth = depth - a * np.sum((reference_xy - center)**2, axis=1)
        water_depth = np.maximum(water_depth, 0.0)
        elevation = water_depth + a * np.sum(xy**2, axis=1)
        return {
            "displacement": displacement,
            "velocity": velocity,
            "points": np.column_stack((xy, elevation)),
            "depth": water_depth,
            "surface_height": elevation,
        }

    builder = ModelBuilder(gravity=gravity, density=1000.0)
    builder.add_water(
        mesh=mesh, bathymetry=ParabolicBottom(ax=a, ay=a),
        surface_height=surface, velocity=(0.0, 0.0),
    )
    minimum_edge_estimate = min(radius / rings, 2 * radius / rings * np.sin(np.pi / sectors))
    return builder.finalize(device=device), {
        "name": "sloshing",
        "dt": 0.1 * minimum_edge_estimate / np.sqrt(gravity * depth),
        "duration": 2 * np.pi / omega,
        "period": 2 * np.pi / omega,
        "omega": omega,
        "amplitude": amplitude,
        "description": "Exact translating wet disk over a paraboloid; polygon shoreline approximation.",
        "analytic": analytic,
        "reference_kind": "nonlinear exact continuum solution, initial material coordinates",
        "reference": "https://doi.org/10.1017/S0022112081001882",
    }
