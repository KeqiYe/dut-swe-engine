"""Command-line cases, optional live views, and final-state export."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


def _positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def _nonnegative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Displacement shallow-water engine")
    p.add_argument("--case", choices=("island", "wave", "sloshing"), default="island")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "warp:cpu"), default="auto")
    p.add_argument("--viewer", choices=("none", "pyvista", "viser"), default="none")
    p.add_argument("--dt", type=_positive, help="simulation time step in seconds")
    p.add_argument("--duration", type=_positive, help="final simulation time in seconds")
    p.add_argument("--steps", type=_nonnegative, help="stop after this many steps (also limits duration)")
    p.add_argument("--refinements", type=_nonnegative, default=0)
    p.add_argument("--fps", type=_positive, default=25.0, help="maximum display update frequency")
    p.add_argument("--speed", type=_positive, default=1.0,
                   help="interactive playback speed in simulation seconds per wall second")
    p.add_argument("--loop", action="store_true",
                   help="replay the interactive case after a short final-frame pause")
    p.add_argument("--vertical-scale", type=_positive, default=1.0,
                   help="display-only vertical scale; does not change the simulation")
    p.add_argument("--field", choices=("eta", "depth", "speed"), default="eta")
    p.add_argument("--clim", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="fixed color range; otherwise taken from initial state")
    p.add_argument("--output", type=Path, help="save the final solver state to NPZ")
    p.add_argument("--host", default="127.0.0.1", help="Viser bind address; defaults to localhost")
    p.add_argument("--port", type=int, default=8080, help="Viser server port")
    p.add_argument("--paused", action="store_true", help="start an interactive viewer paused")
    p.add_argument("--hold", action="store_true", help="keep viewer open at the end; reset remains available")
    p.add_argument("--off-screen", action="store_true", help="PyVista offscreen rendering")
    p.add_argument("--screenshot", type=Path, help="save final PyVista image before closing")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.viewer == "none" and (args.paused or args.hold or args.loop):
        p.error("--paused, --hold, and --loop require an interactive viewer")
    if args.off_screen and (args.viewer != "pyvista" or args.paused or args.hold or args.loop):
        p.error("--off-screen requires PyVista, without --paused, --hold, or --loop")
    if args.loop and args.steps == 0:
        p.error("--loop requires at least one simulation step")
    if args.screenshot and args.viewer != "pyvista":
        p.error("--screenshot requires --viewer pyvista")
    if not 0 <= args.port <= 65535:
        p.error("--port must be between 0 and 65535")
    if args.clim is not None and (not all(math.isfinite(x) for x in args.clim)
                                  or args.clim[0] >= args.clim[1]):
        p.error("--clim must contain finite MIN < MAX")

    from . import cases, run
    from .viewers import create_viewer

    case_fn = {"island": cases.island, "wave": cases.standing_wave,
               "sloshing": cases.sloshing}[args.case]
    viewer = None
    started = time.perf_counter()
    try:
        scale = 2**args.refinements
        mesh_options = {"island": {"refinements": args.refinements},
                        "wave": {"nx": 40 * scale, "ny": 4 * scale},
                        "sloshing": {"rings": 8 * scale, "sectors": 48 * scale}}[args.case]
        model, metadata = case_fn(device=args.device, **mesh_options)
        dt = args.dt if args.dt is not None else float(metadata["dt"])
        duration = args.duration if args.duration is not None else float(metadata["duration"])
        viewer = create_viewer(
            args.viewer, model, field=args.field, paused=args.paused,
            off_screen=args.off_screen, host=args.host, port=args.port,
            clim=tuple(args.clim) if args.clim is not None else None,
            vertical_scale=args.vertical_scale,
        )
        print(f"{metadata.get('description', args.case)}; dt={dt:g} s", flush=True)
        if args.viewer != "none":
            print(f"Playback: {args.speed:g} simulated s / wall s; "
                  f"loop={'on' if args.loop else 'off'}; "
                  f"display vertical scale={args.vertical_scale:g}", flush=True)
        if args.viewer == "viser":
            print(f"Browser viewer: {viewer.url}\n"
                  f"For a remote host: ssh -N -L {viewer.server.get_port()}:127.0.0.1:"
                  f"{viewer.server.get_port()} user@gpu-host", flush=True)
        state = run(
            model, dt=dt, duration=duration,
            viewer=viewer if args.viewer != "none" else None,
            steps=args.steps, fps=args.fps, speed=args.speed,
            loop=args.loop, hold=args.hold,
        )
        if args.screenshot and viewer.is_running:
            viewer.update(state.to_numpy(), model.diagnostics(state))
            viewer.screenshot(args.screenshot)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            state.save(args.output)
        report = {"case": args.case, "time": float(state.time),
                  "steps": int(state.step_index), "wall_seconds": time.perf_counter() - started,
                  "diagnostics": model.diagnostics(state)}
        print(json.dumps(report, indent=2, default=float))
        return 0
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    except (ImportError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"swe: {exc}", file=sys.stderr)
        return 1
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
