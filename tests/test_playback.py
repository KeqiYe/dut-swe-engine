"""Playback timing uses a fake clock: tests never sleep or need a GUI."""

from concurrent.futures import Future

import pytest

from swe import cli, runtime
from swe.viewers import NullViewer


class Clock:
    now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ImmediateExecutor:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        result = Future()
        result.set_result(function(*args))
        return result


class State:
    time = 0.0
    step_index = 0

    def to_numpy(self):
        return {"time": self.time}


class Model:
    allocations = 0

    def state(self):
        self.allocations += 1
        return State()

    def diagnostics(self, state):
        return {}


class Solver:
    def __init__(self, clock):
        self.clock = clock
        self.steps = []

    def step(self, state, spare, dt):
        self.steps.append((state.step_index, dt, self.clock.now))
        spare.time = state.time + dt
        spare.step_index = state.step_index + 1


class Viewer(NullViewer):
    def __init__(self, clock, *, paused=False, on_poll=lambda viewer: None):
        super().__init__(paused=paused)
        self.clock = clock
        self.on_poll = on_poll
        self.frames = []
        self.polls = 0

    def poll(self):
        self.polls += 1
        assert self.polls < 20000, "playback did not terminate"
        self.on_poll(self)

    def update(self, snapshot, diagnostics):
        self.frames.append((snapshot["time"], self.clock.now))


@pytest.fixture
def playback(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(runtime, "time", clock)
    monkeypatch.setattr(runtime, "ThreadPoolExecutor", ImmediateExecutor)

    def run(*, speed=1.0, loop=False, paused=False, duration=1.0,
            dt=0.1, steps=None, on_poll=lambda viewer: None):
        model = Model()
        solver = Solver(clock)
        viewer = Viewer(clock, paused=paused, on_poll=on_poll)
        monkeypatch.setattr(runtime, "DisplacementSolver", lambda model: solver)
        result = runtime.run(
            model, viewer=viewer, dt=dt, duration=duration, speed=speed,
            loop=loop, fps=25.0, steps=steps,
        )
        return result, model, solver, viewer, clock

    return run


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_playback_is_paced_without_changing_physics_steps(playback, speed):
    result, _, solver, viewer, clock = playback(speed=speed, dt=0.2, duration=0.75)
    assert result.time == pytest.approx(0.75)
    assert [step[1] for step in solver.steps] == pytest.approx([0.2, 0.2, 0.2, 0.15])
    for (_, _, wall_time), simulated_time in zip(solver.steps, [0.2, 0.4, 0.6, 0.75]):
        assert wall_time >= simulated_time / speed - 1e-10
    assert clock.now == pytest.approx(0.75 / speed, abs=0.006)
    assert viewer.frames[-1][0] == pytest.approx(0.75)


def test_resume_does_not_catch_up_time_spent_paused(playback):
    paused = resumed = False

    def control(viewer):
        nonlocal paused, resumed
        if viewer.clock.now >= 0.21 and not paused:
            viewer.paused = True
            paused = True
        elif viewer.clock.now >= 2.21 and not resumed:
            viewer.paused = False
            resumed = True

    result, _, solver, _, clock = playback(on_poll=control)
    assert result.time == pytest.approx(1.0)
    assert solver.steps[2][2] >= 2.31 - 1e-10
    assert clock.now == pytest.approx(3.01, abs=0.01)


def test_single_step_and_reset_while_paused_then_resume(playback):
    actions = set()

    def control(viewer):
        for wall_time, name, action in (
            (0.1, "step", viewer.request_step),
            (0.2, "reset", viewer.request_reset),
            (0.3, "second_step", viewer.request_step),
            (0.6, "resume", lambda: setattr(viewer, "paused", False)),
        ):
            if viewer.clock.now >= wall_time and name not in actions:
                actions.add(name)
                action()

    result, model, solver, viewer, clock = playback(
        paused=True, duration=0.5, on_poll=control,
    )
    assert result.time == pytest.approx(0.5)
    assert model.allocations == 4
    assert [step[0] for step in solver.steps] == [0, 0, 1, 2, 3, 4]
    assert solver.steps[2][2] >= 0.7 - 1e-10
    assert clock.now == pytest.approx(1.0, abs=0.008)
    assert any(t == 0.0 and wall >= 0.2 for t, wall in viewer.frames)


def test_loop_shows_final_frame_then_restarts_from_initial_state(playback):
    def control(viewer):
        # Final states are refreshed during the dwell, so count the second
        # initial state and then its final state rather than repeated frames.
        restarted = any(t == 0.0 and wall > 0.3 for t, wall in viewer.frames)
        if restarted and viewer.frames[-1][0] >= 0.3 - 1e-10:
            viewer.close()

    result, model, solver, viewer, _ = playback(loop=True, duration=0.3, on_poll=control)
    assert result.time == pytest.approx(0.3)
    assert model.allocations == 4
    assert [step[0] for step in solver.steps] == [0, 1, 2, 0, 1, 2]
    assert [step[1] for step in solver.steps] == pytest.approx([0.1] * 6)
    first_finish = next(wall for t, wall in viewer.frames if t >= 0.3 - 1e-10)
    second_start = next(wall for t, wall in viewer.frames if t == 0.0 and wall > 0.3)
    assert second_start - first_finish >= 0.5
    assert not viewer.paused


def test_loop_respects_step_limit(playback):
    def control(viewer):
        if viewer.clock.now > 0.85:
            viewer.close()

    _, _, solver, _, _ = playback(loop=True, steps=1, duration=1.0, on_poll=control)
    assert [step[0] for step in solver.steps] == [0, 0]


@pytest.mark.parametrize("flag", ["--speed", "--vertical-scale"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_playback_options_require_positive_finite_values(flag, value):
    with pytest.raises(SystemExit):
        cli.parser().parse_args([flag, value])


def test_headless_stays_unpaced(playback, monkeypatch):
    # This fixture installs the fake clock and executor without running a viewer.
    model = Model()
    clock = Clock()
    solver = Solver(clock)
    monkeypatch.setattr(runtime, "DisplacementSolver", lambda model: solver)
    result = runtime.run(model, dt=0.1, duration=1.0)
    assert result.time == pytest.approx(1.0)
    assert clock.now == 0.0
