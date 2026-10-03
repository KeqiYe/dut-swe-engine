"""Public runner accepts application-built models without importing a demo."""

import numpy as np
import pytest

import swe
from swe.viewers import NullViewer


@pytest.fixture
def model():
    mesh = swe.Mesh.rectangle(8, 2)
    return swe.ModelBuilder(gravity=9.81).add_water(
        mesh=mesh, bathymetry=swe.FlatBottom(0.0),
        surface_height=lambda p: 1.0 + 0.001 * np.cos(np.pi * p[:, 0] / 10.0),
    ).finalize(device="cpu")


def test_public_run_matches_manual_solver_loop_and_preserves_initial_model(model):
    expected, spare = model.state(), model.state()
    solver = swe.DisplacementSolver(model)
    for dt in [0.01] * 10 + [0.005]:
        solver.step(expected, spare, dt)
        expected, spare = spare, expected

    result = swe.run(model, dt=0.01, duration=0.105)
    assert isinstance(result, swe.State)
    assert result.time == pytest.approx(0.105)
    assert result.step_index == 11
    np.testing.assert_allclose(result.displacement, expected.displacement, atol=1e-17)
    np.testing.assert_allclose(result.velocity, expected.velocity, atol=1e-17)
    assert model.state().time == 0.0
    assert np.all(model.state().displacement == 0.0)
    repeated = swe.run(model, dt=0.01, duration=0.105)
    np.testing.assert_array_equal(result.displacement, repeated.displacement)


@pytest.mark.parametrize("steps, expected_time", [(0, 0.0), (3, 0.03), (np.int64(2), 0.02)])
def test_public_step_limit(model, steps, expected_time):
    state = swe.run(model, dt=0.01, duration=0.105, steps=steps)
    assert state.time == pytest.approx(expected_time)
    assert state.step_index == steps


@pytest.mark.parametrize("name", ["dt", "duration", "fps", "speed"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True, None])
def test_invalid_positive_parameters_fail_before_allocating_state(name, value):
    options = {"dt": 0.01, "duration": 0.1, name: value}
    with pytest.raises(ValueError, match=name):
        swe.run(None, **options)


@pytest.mark.parametrize("steps", [-1, 1.5, "2", True, float("nan")])
def test_invalid_step_limit_fails_before_allocating_state(steps):
    with pytest.raises(ValueError, match="steps"):
        swe.run(None, dt=0.01, duration=0.1, steps=steps)


@pytest.mark.parametrize("option", ["loop", "hold"])
def test_headless_rejects_interactive_controls(option):
    with pytest.raises(ValueError, match="viewer"):
        swe.run(None, dt=0.01, duration=0.1, **{option: True})


def test_loop_rejects_empty_run():
    with pytest.raises(ValueError, match="step"):
        swe.run(None, dt=0.01, duration=0.1, viewer=NullViewer(), loop=True, steps=0)


def test_caller_retains_ownership_of_viewer(model):
    viewer = NullViewer()
    state = swe.run(model, dt=0.01, duration=0.1, viewer=viewer, steps=0)
    assert state.step_index == 0
    assert viewer.is_running
    assert viewer.paused
    viewer.close()


def test_small_positive_duration_still_advances(model):
    state = swe.run(model, dt=1e-15, duration=1e-14)
    assert state.time == pytest.approx(1e-14, abs=1e-28)
    assert state.step_index == 10
