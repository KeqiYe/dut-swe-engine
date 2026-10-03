"""Standalone refined-island application using the installed ``swe`` package.

Install from the project root first: ``python -m pip install -e .``
For a paced, looping browser animation with ten times as many triangles,
install ``-e '.[web]'`` and run::

    python examples/island_refined.py --device cpu --mesh-factor 10 \
        --viewer viser --hide-nodes --wireframe --loop --speed 0.2

The original 547-node, 1006-triangle mesh and physical scene reproduce the
MATLAB island benchmark. Integer mesh factors count triangles, not points
along each coordinate axis. Extra interior splits retain a conforming mesh;
boundary resolution follows the preceding uniform 1-to-4 refinement level.
This example defines its scene itself, without internal benchmark factories.
"""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
import math
import sys

import numpy as np
import swe


def _edges(triangles):
    return np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                   triangles[:, [2, 0]])), axis=1)


def _project_shoreline(mesh, bottom, surface):
    """Move curved-shore boundary nodes onto the wet side of the zero-depth contour."""
    edges, count = np.unique(_edges(mesh.triangles), axis=0, return_counts=True)
    boundary = np.unique(edges[count == 1])
    shoreline = boundary[~mesh.fixed[boundary].any(axis=1)]
    center = np.asarray(bottom.center)
    rays = mesh.points[shoreline] - center
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    low, high = np.zeros(len(shoreline)), np.full(len(shoreline), 8 * bottom.sigma)
    for _ in range(60):
        middle = (low + high) / 2
        xy = center + middle[:, None] * rays
        wet = surface(xy) >= bottom.evaluate(xy)[0]
        high = np.where(wet, middle, high)
        low = np.where(wet, low, middle)
    points = mesh.points.copy()
    points[shoreline] = center + high[:, None] * rays
    return swe.Mesh(points, mesh.triangles, mesh.fixed)


def _refine_interior_to(mesh, target):
    """Bisect both neighbors of long interior edges without hanging nodes.

    Each split adds one vertex and two triangles. Within a batch no triangle
    is split twice. Boundary edges stay at the preceding uniform refinement
    level, retaining the projected shore and the impermeable wall constraints.
    """
    while len(mesh.triangles) < target:
        triangles = mesh.triangles
        edges, inverse, count = np.unique(_edges(triangles), axis=0,
                                          return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind="stable")
        first = np.r_[0, np.cumsum(count[:-1])]
        interior = np.flatnonzero(count == 2)
        length2 = np.sum(np.diff(mesh.points[edges[interior]], axis=1)[:, 0] ** 2, axis=1)
        candidates = interior[np.argsort(-length2, kind="stable")]
        used = np.zeros(len(triangles), dtype=bool)
        selected = []
        for edge in candidates:
            neighbors = order[first[edge]:first[edge] + 2] % len(triangles)
            if not used[neighbors].any():
                selected.append((edge, neighbors))
                used[neighbors] = True
                if len(selected) == (target - len(triangles)) // 2:
                    break
        if not selected:
            raise ValueError("cannot reach the requested triangle count with interior bisection")
        new_points, children = [], []
        for edge, neighbors in selected:
            a, b = edges[edge]
            midpoint = len(mesh.points) + len(new_points)
            new_points.append((mesh.points[a] + mesh.points[b]) / 2)
            for index in neighbors:
                tri = triangles[index]
                for local in range(3):
                    start, end = tri[local], tri[(local + 1) % 3]
                    if {start, end} == {a, b}:
                        opposite = tri[(local + 2) % 3]
                        children.extend(((start, midpoint, opposite),
                                         (midpoint, end, opposite)))
                        break
        mesh = swe.Mesh(
            np.concatenate((mesh.points, new_points)),
            np.concatenate((triangles[~used], np.asarray(children, dtype=np.int32))),
            np.concatenate((mesh.fixed, np.zeros((len(new_points), 2), dtype=bool))),
        )
    return mesh


def build_model(device="auto", gravity=9.81, mesh_factor=1):
    """Build an integer multiple of the original triangle count.

    Uniform 1-to-4 subdivision is followed by conforming interior bisection
    when necessary. The mesh factor describes triangle count, not each axis.
    """
    gravity = float(gravity)
    if not math.isfinite(gravity) or gravity <= 0:
        raise ValueError("gravity must be finite and positive")
    if isinstance(mesh_factor, bool) or not isinstance(mesh_factor, int) or mesh_factor < 1:
        raise ValueError("mesh_factor must be a positive integer")

    # Packaged reference data work after an ordinary wheel installation too.
    # Supply your own arrays here to replace this mesh: coordinates (N, 2),
    # counterclockwise triangle indices (E, 3), and fixed-DOF mask (N, 2).
    with files("swe").joinpath("data/island_mesh.npz").open("rb") as stream:
        with np.load(stream, allow_pickle=False) as data:
            mesh = swe.Mesh(data["points"], data["triangles"], data["fixed"])
    original_triangles = len(mesh.triangles)

    # Bed elevations and surface heights are in metres, measured from z = 0.
    bottom = swe.GaussianBottom(height=0.875, center=(12.5, 15.0), sigma=1.6)
    mean_height, amplitude = 0.32, 0.032
    k = np.sqrt(3 * amplitude / (4 * mean_height**2 * (mean_height + amplitude)))
    wave_center = 5 / k
    c = np.sqrt(gravity * (mean_height + amplitude))

    def profile(xy):
        return 1 / np.cosh(k * (xy[:, 0] - wave_center)) ** 2

    def initial_surface(xy):
        return mean_height + amplitude * profile(xy)

    def initial_velocity(xy):
        velocity = np.zeros_like(xy, dtype=float)
        velocity[:, 0] = c * k * np.sqrt(4 * mean_height * amplitude / 3) * profile(xy)
        return velocity

    target = original_triangles * mesh_factor
    while 4 * len(mesh.triangles) <= target:
        mesh = _project_shoreline(mesh.refined(), bottom, initial_surface)
    mesh = _refine_interior_to(mesh, target)

    builder = swe.ModelBuilder(gravity=gravity, density=1000.0)
    builder.add_water(
        mesh=mesh,
        bathymetry=bottom,
        surface_height=initial_surface,
        velocity=initial_velocity,
    )
    model = builder.finalize(device=device)
    time_scale = np.sqrt(9.81 / gravity)
    step_divisor = 2 ** math.ceil(math.log2(math.sqrt(mesh_factor)))
    return model, {
        "dt": 0.00625 / step_divisor * time_scale,
        "duration": (10.0 if mesh_factor == 1 else 3.0) * time_scale,
        "mesh_factor": mesh_factor, "nodes": len(mesh.points),
        "triangles": len(mesh.triangles), "original_triangles": original_triangles,
    }


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def positive_integer(value):
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gravity", type=positive, default=9.81, help="gravity in m/s²")
    parser.add_argument("--mesh-factor", type=positive_integer, default=1,
                        help="integer multiple of the original 1006 triangles")
    parser.add_argument("--viewer", choices=("none", "viser", "pyvista"), default="none")
    parser.add_argument("--dt", type=positive, help="physics time step in seconds")
    parser.add_argument("--duration", type=positive, help="simulation duration in seconds")
    parser.add_argument("--speed", type=positive, default=1.0, help="simulated seconds per wall second")
    parser.add_argument("--loop", action="store_true", help="repeat the interactive animation")
    parser.add_argument("--hold", action="store_true", help="keep the viewer open after completion")
    parser.add_argument("--port", type=int, default=8080, help="local browser viewer port")
    parser.add_argument("--vertical-scale", type=positive, default=1.0, help="display-only height scale")
    parser.add_argument("--hide-nodes", action="store_true", help="hide the nodal point overlay")
    parser.add_argument("--wireframe", action="store_true", help="show triangle edges")
    args = parser.parse_args(argv)
    if args.viewer == "none" and (args.loop or args.hold):
        parser.error("--loop and --hold require an interactive viewer")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")

    viewer = None
    try:
        model, settings = build_model(device=args.device, gravity=args.gravity,
                                      mesh_factor=args.mesh_factor)
        dt = args.dt if args.dt is not None else settings["dt"]
        duration = args.duration if args.duration is not None else settings["duration"]
        if args.viewer != "none":
            from swe.viewers import create_viewer

            viewer = create_viewer(
                args.viewer, model, host="127.0.0.1", port=args.port,
                vertical_scale=args.vertical_scale,
                show_nodes=not args.hide_nodes, wireframe=args.wireframe,
            )
            if args.viewer == "viser":
                print(f"Browser viewer: {viewer.url}", flush=True)

        print(f"Island: {settings['nodes']} nodes, {settings['triangles']} triangles "
              f"({args.mesh_factor}× original), dt={dt:g} s, duration={duration:g} s", flush=True)
        state = swe.run(
            model, dt=dt, duration=duration, viewer=viewer,
            speed=args.speed, loop=args.loop, hold=args.hold,
        )
        print(json.dumps({
            "case": "island", "time": state.time, "steps": state.step_index,
            "diagnostics": model.diagnostics(state),
        }, indent=2, default=float))
        return 0
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (ImportError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"island: {exc}", file=sys.stderr)
        return 1
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
