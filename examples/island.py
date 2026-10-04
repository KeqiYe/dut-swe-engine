"""Standalone island application using the installed ``swe`` package.

Install from the project root first: ``python -m pip install -e .``
Then run: ``python examples/island.py --device cpu``
For a paced, looping browser animation, install ``-e '.[web]'`` and run::

    python examples/island.py --device cpu --viewer viser --loop --speed 0.5

The base island benchmark uses a 547-node, 1006-triangle mesh.
This example defines its scene itself; it does not
call the package's built-in benchmark factories or command-line application.
"""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
import math
import sys

import numpy as np
import swe


def build_model(device="auto", gravity=9.81):
    """Build the base island scene and return its model and time settings."""
    gravity = float(gravity)
    if not math.isfinite(gravity) or gravity <= 0:
        raise ValueError("gravity must be finite and positive")

    # Packaged reference data work after an ordinary wheel installation too.
    # Supply your own arrays here to replace this mesh: coordinates (N, 2),
    # counterclockwise triangle indices (E, 3), and fixed-DOF mask (N, 2).
    with files("swe").joinpath("data/island_mesh.npz").open("rb") as stream:
        with np.load(stream, allow_pickle=False) as data:
            mesh = swe.Mesh(data["points"], data["triangles"], data["fixed"])

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

    builder = swe.ModelBuilder(gravity=gravity, density=1000.0)
    builder.add_water(
        mesh=mesh,
        bathymetry=bottom,
        surface_height=initial_surface,
        velocity=initial_velocity,
    )
    model = builder.finalize(device=device)
    time_scale = np.sqrt(9.81 / gravity)
    return model, {"dt": 0.00625 * time_scale, "duration": 10.0 * time_scale}


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gravity", type=positive, default=9.81, help="gravity in m/s²")
    parser.add_argument("--viewer", choices=("none", "viser", "pyvista"), default="none")
    parser.add_argument("--dt", type=positive, help="physics time step in seconds")
    parser.add_argument("--duration", type=positive, help="simulation duration in seconds")
    parser.add_argument("--speed", type=positive, default=1.0, help="simulated seconds per wall second")
    parser.add_argument("--loop", action="store_true", help="repeat the interactive animation")
    parser.add_argument("--hold", action="store_true", help="keep the viewer open after completion")
    parser.add_argument("--port", type=int, default=8080, help="local browser viewer port")
    parser.add_argument("--vertical-scale", type=positive, default=1.0, help="display-only height scale")
    args = parser.parse_args(argv)
    if args.viewer == "none" and (args.loop or args.hold):
        parser.error("--loop and --hold require an interactive viewer")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")

    viewer = None
    try:
        model, settings = build_model(device=args.device, gravity=args.gravity)
        dt = args.dt if args.dt is not None else settings["dt"]
        duration = args.duration if args.duration is not None else settings["duration"]
        if args.viewer != "none":
            from swe.viewers import create_viewer

            viewer = create_viewer(
                args.viewer, model, host="127.0.0.1", port=args.port,
                vertical_scale=args.vertical_scale,
            )
            if args.viewer == "viser":
                print(f"Browser viewer: {viewer.url}", flush=True)

        print(f"Base island: dt={dt:g} s, duration={duration:g} s", flush=True)
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
