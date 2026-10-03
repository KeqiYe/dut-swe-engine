"""Standalone application: a parabolic water basin using the ``swe`` package.

Install from the repo root with ``python -m pip install -e '.[web]'``.
Headless: ``python examples/sloshing.py --device cpu``.
Animation: ``python examples/sloshing.py --device cpu --viewer viser --loop
--speed 0.5 --vertical-scale 5`` (one shell command).
"""

import argparse

import numpy as np
import swe


def basin_mesh(radius, center, rings, sectors):
    """An application-supplied triangular mesh of the initial wet disk."""
    if not isinstance(rings, int) or not isinstance(sectors, int) or rings < 1 or sectors < 8:
        raise ValueError("rings must be a positive integer and sectors an integer >= 8")
    angles = 2 * np.pi * np.arange(sectors) / sectors
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    points = [np.asarray(center, dtype=float)]
    for ring in range(1, rings + 1):
        points.extend(center + radius * ring / rings * circle)
    triangles = []
    for sector in range(sectors):
        triangles.append((0, 1 + sector, 1 + (sector + 1) % sectors))
    for ring in range(1, rings):
        inner, outer = 1 + (ring - 1) * sectors, 1 + ring * sectors
        for sector in range(sectors):
            following = (sector + 1) % sectors
            triangles.extend([
                (inner + sector, outer + sector, outer + following),
                (inner + sector, outer + following, inner + following),
            ])
    # Mesh accepts external horizontal nodes, CCW triangles and optional walls.
    return swe.Mesh(np.asarray(points), np.asarray(triangles, dtype=np.int32))


def build_model(
    *, device="auto", radius=10.0, depth=1.0, amplitude=0.5,
    gravity=9.81, rings=8, sectors=48,
):
    if not np.all(np.isfinite([radius, depth, amplitude, gravity])):
        raise ValueError("scene parameters must be finite")
    if min(radius, depth, gravity) <= 0:
        raise ValueError("radius, depth and gravity must be positive")
    curvature = depth / radius**2
    mesh = basin_mesh(radius, np.array([amplitude, 0.0]), rings, sectors)

    def initial_surface(xy):
        return depth + 2 * curvature * amplitude * xy[:, 0] - curvature * amplitude**2

    builder = swe.ModelBuilder(gravity=gravity, density=1000.0)
    builder.add_water(
        mesh=mesh,
        bathymetry=swe.ParabolicBottom(ax=curvature, ay=curvature),
        surface_height=initial_surface,
        velocity=(0.0, 0.0),
    )
    model = builder.finalize(device=device)
    omega = np.sqrt(2 * gravity * curvature)
    edge = min(radius / rings, 2 * radius / rings * np.sin(np.pi / sectors))
    return model, {"dt": 0.1 * edge / np.sqrt(gravity * depth), "duration": 2 * np.pi / omega}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--viewer", choices=("none", "viser", "pyvista"), default="none")
    parser.add_argument("--dt", type=float, help="physics time step in seconds")
    parser.add_argument("--duration", type=float, help="simulation duration in seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vertical-scale", type=float, default=1.0)
    args = parser.parse_args()

    # These scene parameters are application inputs, not engine internals.
    model, settings = build_model(device=args.device)
    viewer = None
    try:
        if args.viewer != "none":
            from swe.viewers import create_viewer

            viewer = create_viewer(
                args.viewer, model, port=args.port, vertical_scale=args.vertical_scale,
            )
            if args.viewer == "viser":
                print(f"Open {viewer.url}", flush=True)
        state = swe.run(
            model,
            dt=settings["dt"] if args.dt is None else args.dt,
            duration=settings["duration"] if args.duration is None else args.duration,
            viewer=viewer, speed=args.speed, loop=args.loop, hold=args.hold,
        )
        print(model.diagnostics(state))
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
