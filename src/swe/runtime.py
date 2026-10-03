"""Generic simulation runner shared by applications and the command line.

No case selection or graphics dependency is needed to import this module.
Viewers, when supplied, are created and closed by the calling application.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from numbers import Integral
import time
from typing import TYPE_CHECKING

from .core import DisplacementSolver, Model, State

if TYPE_CHECKING:
    from .viewers import NullViewer


def _positive(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def run(
    model: Model, *, dt: float, duration: float,
    viewer: NullViewer | None = None, steps: int | None = None,
    fps: float = 25.0, speed: float = 1.0,
    loop: bool = False, hold: bool = False,
) -> State:
    """Run a finalized model from its initial state and return the final state.

    ``dt`` is the physical step in seconds. The last step is shortened to land
    on ``duration``; ``steps`` optionally stops the run earlier. With no viewer,
    computation proceeds as fast as the selected CPU/CUDA backend permits.

    A supplied viewer is polled on the calling thread and receives snapshots at
    up to ``fps``. Physics runs in a worker, paced at ``speed`` simulation seconds
    per wall second. Pause, single-step and reset requests are honored. ``loop``
    restarts from the model's initial state after a half-second final-frame pause;
    ``hold`` keeps a finished viewer responsive until it is closed. Both require
    a viewer. The caller owns its lifetime and must call ``viewer.close()``.

    Playback settings never change the physical time step. Numerical errors
    propagate to the caller. For a custom time loop or checkpoint continuation,
    use :class:`DisplacementSolver` directly instead.
    """
    dt = _positive("dt", dt)
    duration = _positive("duration", duration)
    fps = _positive("fps", fps)
    speed = _positive("speed", speed)
    if steps is not None and (isinstance(steps, bool)
                              or not isinstance(steps, Integral) or steps < 0):
        raise ValueError("steps must be a nonnegative integer or None")
    if viewer is None and (loop or hold):
        raise ValueError("loop and hold require a viewer")
    if loop and steps == 0:
        raise ValueError("loop requires at least one simulation step")

    state, spare = model.state(), model.state()
    solver = DisplacementSolver(model)
    if viewer is None:
        return _run_headless(solver, state, spare, dt, duration, steps)
    return _run_interactive(
        model, solver, state, spare, viewer, dt=dt, duration=duration,
        steps=steps, fps=fps, speed=speed, loop=loop, hold=hold,
    )


def _done(state, duration: float, steps: int | None) -> bool:
    return (state.time >= duration - 1e-12 * duration
            or (steps is not None and state.step_index >= steps))


def _run_headless(solver, state, spare, dt, duration, steps):
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


def _run_interactive(model, solver, state, spare, viewer, *, dt, duration,
                     steps, fps, speed, loop, hold):
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
                finished = _done(state, duration, steps)
                if finished and not loop:
                    viewer.paused = True
                if now >= next_display or (finished and displayed_time != state.time):
                    if displayed_time != state.time:
                        snapshot, diagnostics = state.to_numpy(), model.diagnostics(state)
                    viewer.update(snapshot, diagnostics)
                    displayed_time = state.time
                    next_display = now + 1.0 / fps
                if finished:
                    # In loop mode keep the running/paused control under the
                    # user's control and show the final state for half a second.
                    if loop:
                        if finished_at is None or viewer.paused:
                            finished_at = now
                        elif now - finished_at >= 0.5:
                            state, spare = model.state(), model.state()
                            displayed_time = None
                            next_display = 0.0
                            finished_at = None
                            reanchor = True
                    elif not hold:
                        break
                    time.sleep(0.005)
                    continue
                single_step = viewer.consume_step_request()
                target_time = simulation_origin + (now - clock_origin) * speed
                # Only schedule whole physics steps that are due. The last
                # step may be shorter to land exactly on the requested duration.
                available = math.floor((target_time - state.time) / dt + 1e-10)
                if target_time >= duration:
                    available = max(available, math.ceil((duration - state.time) / dt))
                count = 1 if single_step else min(32, max(0, available))
                if count and (not viewer.paused or single_step):
                    pending = worker.submit(
                        _advance_chunk, solver, state, spare, dt, duration,
                        steps, count,
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


