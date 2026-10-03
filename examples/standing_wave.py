"""Standalone application: build and simulate a smooth wave with ``import swe``.

Install the engine first: ``python -m pip install -e .`` (from the repo root).
Then run ``python examples/standing_wave.py --device cpu``.
Optional animation: add ``--viewer viser --loop --speed 0.5``.
This file can also be copied into another project using the installed engine.
"""

import argparse

import numpy as np
import swe


def build_model(
    *, device="auto", nx=40, ny=4, length=10.0, width=1.0,
    depth=1.0, amplitude=0.001, gravity=9.81,
):
    """Inputs belong to this application; the engine owns numerical stepping."""
    if not np.all(np.isfinite([length, width, depth, gravity, amplitude])):
        raise ValueError("scene parameters must be finite")
    if min(length, width, depth, gravity) <= 0 or abs(amplitude) >= depth:
        raise ValueError("dimensions, depth and gravity must be positive; |amplitude| < depth")
    mesh = swe.Mesh.rectangle(nx=nx, ny=ny, size=(length, width), walls=True)

    def water_surface(xy):
        return depth + amplitude * np.cos(np.pi * xy[:, 0] / length)

    builder = swe.ModelBuilder(gravity=gravity, density=1000.0)
    builder.add_water(
        mesh=mesh,
        bathymetry=swe.FlatBottom(z=0.0),
        surface_height=water_surface,
        velocity=(0.0, 0.0),
    )
    model = builder.finalize(device=device)
    # Small-amplitude period and a conservative initial time step for this mesh.
    dt = 0.1 * min(length / nx, width / ny) / np.sqrt(gravity * (depth + abs(amplitude)))
    return model, {"dt": dt, "duration": 2 * length / np.sqrt(gravity * depth)}


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

    # Change build_model's arguments here to define your own water problem.
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
        # swe.run is a convenience runner; basic.py shows an explicit step loop.
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
