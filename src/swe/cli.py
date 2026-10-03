"""Command-line cases, optional live views, and final-state export."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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


def _done(state, duration: float, steps: int | None) -> bool:
    return (state.time >= duration - 1e-12 * max(1.0, duration)
            or (steps is not None and state.step_index >= steps))


def _run_headless(model, solver, state, spare, dt, duration, steps):
    while not _done(state, duration, steps):
        solver.step(state, spare, min(dt, duration - state.time))
        state, spare = spare, state
    return state


def _advance_chunk(solver, state, spare, dt, duration, steps,
                   max_steps=32, wall_budget=0.005):
    """Bound a worker batch by both step count and elapsed host time.

    A single slow solver step cannot be interrupted, but it never blocks the
    UI thread. All state reads happen after this batch returns its two buffers.
    """
    deadline = time.perf_counter() + wall_budget
    for _ in range(max_steps):
        if _done(state, duration, steps):
            break
        solver.step(state, spare, min(dt, duration - state.time))
        state, spare = spare, state
        if time.perf_counter() >= deadline:
            break
    return state, spare


def _run_interactive(model, solver, state, spare, viewer, args, dt, duration):
    # The only solver worker owns each in-flight bounded batch. UI operations
    # stay on the main thread, even during slow CPU steps / initial CUDA compile.
    # Snapshots and diagnostics are read only after a batch has completed.
    next_display = 0.0
    pending = None
    reset_pending = False
    displayed_time = None
    snapshot = diagnostics = None
    clock_origin = time.perf_counter()
    simulation_origin = state.time
    was_paused = viewer.paused
    reanchor = False
    finished_at = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="swe-step") as worker:
        try:
            while viewer.is_running:
                viewer.poll()
                if not viewer.is_running:
                    break
                reset_pending |= viewer.consume_reset_request()
                # Observe pause transitions even while a worker owns the state.
                # Resuming starts from the latest completed state, without a
                # burst of steps for wall time spent paused.
                if viewer.paused != was_paused:
                    reanchor = True
                    was_paused = viewer.paused
                if pending is not None:
                    if not pending.done():
                        time.sleep(0.002)
                        continue
                    state, spare = pending.result()
                    pending = None

                if reset_pending:
                    state, spare = model.state(), model.state()
                    reset_pending = False
                    displayed_time = None
                    next_display = 0.0
                    finished_at = None
                    reanchor = True

                now = time.perf_counter()
                if reanchor or viewer.paused:
                    clock_origin, simulation_origin = now, state.time
                    reanchor = False
                finished = _done(state, duration, args.steps)
                if finished and not args.loop:
                    viewer.paused = True
                if now >= next_display or (finished and displayed_time != state.time):
                    if displayed_time != state.time:
                        snapshot, diagnostics = state.to_numpy(), model.diagnostics(state)
                    viewer.update(snapshot, diagnostics)
                    displayed_time = state.time
                    next_display = now + 1.0 / args.fps
                if finished:
                    # In loop mode keep the running/paused control under the
                    # user's control and show the final state for half a second.
                    if args.loop:
                        if finished_at is None or viewer.paused:
                            finished_at = now
                        elif now - finished_at >= 0.5:
                            state, spare = model.state(), model.state()
                            displayed_time = None
                            next_display = 0.0
                            finished_at = None
                            reanchor = True
                    elif not args.hold:
                        break
                    time.sleep(0.005)
                    continue
                single_step = viewer.consume_step_request()
                target_time = simulation_origin + (now - clock_origin) * args.speed
                # Only schedule whole physics steps that are due. The last
                # step may be shorter to land exactly on the requested duration.
                available = math.floor((target_time - state.time) / dt + 1e-10)
                if target_time >= duration:
                    available = max(available, math.ceil((duration - state.time) / dt))
                count = 1 if single_step else min(32, max(0, available))
                if count and (not viewer.paused or single_step):
                    pending = worker.submit(
                        _advance_chunk, solver, state, spare, dt, duration,
                        args.steps, count,
                    )
                    if single_step:
                        reanchor = True
                else:
                    time.sleep(0.002)
        finally:
            # Do not close the CUDA/GUI resources while a state write is active.
            if pending is not None:
                state, spare = pending.result()
    return state


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

    from . import DisplacementSolver, cases
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
        state, spare = model.state(), model.state()
        solver = DisplacementSolver(model)
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
                  f"For delltower: ssh -N -L {viewer.server.get_port()}:127.0.0.1:"
                  f"{viewer.server.get_port()} delltower", flush=True)
        if args.viewer == "none":
            state = _run_headless(model, solver, state, spare, dt, duration, args.steps)
        else:
            state = _run_interactive(model, solver, state, spare, viewer, args, dt, duration)
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
