"""Physics checks independent of the MATLAB regression data."""

import numpy as np
import pytest

from swe import DisplacementSolver
from swe.cases import island, sloshing, standing_wave


def evolve(model, duration, dt):
    """Land on the requested time exactly with a constant time step."""
    solver = DisplacementSolver(model)
    state, scratch = model.state(), model.state()
    steps = int(np.ceil(duration / dt))
    for _ in range(steps):
        solver.step(state, scratch, duration / steps)
        state, scratch = scratch, state
    return state


def relative_energy_change(model, before, after):
    e0 = model.diagnostics(before)["energy"]
    return abs(model.diagnostics(after)["energy"] - e0) / abs(e0)


def test_island_matches_original_configuration():
    model, meta = island(device="cpu")
    initial = model.state().to_numpy()
    assert initial["points"].shape == (547, 3)
    assert meta["dt"] == 0.00625
    assert meta["duration"] == 10.0
    assert np.isfinite(initial["velocity"]).all()
    # A short integration catches sign and geometry mistakes in the terrain.
    state = evolve(model, duration=0.05, dt=meta["dt"])
    diagnostic = model.diagnostics(state)
    assert diagnostic["min_j"] > 0.95
    assert diagnostic["min_depth"] > 0.0
    assert relative_energy_change(model, model.state(), state) < 1e-6


@pytest.mark.parametrize("refinements", [1, 3])
def test_island_refinement_projects_shoreline_to_wet_contour(refinements):
    model, _ = island(device="cpu", refinements=refinements)
    assert len(model.triangles) == 1006 * 4**refinements
    assert np.min(model.depth0) > 0.0
    assert np.min(model.node_depth0) > -1e-12
    # Confirm the requested large scene can advance, not just be constructed.
    state = evolve(model, duration=0.0005, dt=0.0005)
    assert model.diagnostics(state)["min_j"] > 0.99


def test_standing_wave_rest_state_remains_stationary():
    model, meta = standing_wave(device="cpu", nx=12, ny=3, amplitude=0.0)
    state = evolve(model, duration=0.2, dt=meta["dt"])
    result = state.to_numpy()
    np.testing.assert_allclose(result["velocity"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["displacement"], 0.0, atol=1e-12)


def test_standing_wave_quarter_period_velocity():
    model, meta = standing_wave(device="cpu", nx=32, ny=4, amplitude=1e-4)
    initial = model.state()
    state = evolve(model, duration=meta["period"] / 4, dt=meta["dt"])
    result = state.to_numpy()
    expected = meta["analytic"](result["points"][:, :2], result["time"])
    relative_error = np.linalg.norm(result["velocity"] - expected["velocity"]) / np.linalg.norm(expected["velocity"])
    assert relative_error < 0.04
    assert relative_energy_change(model, initial, state) < 1e-6
    assert model.diagnostics(state)["min_j"] > 0.99


def test_standing_wave_second_order_time_convergence():
    model, _ = standing_wave(device="cpu", nx=12, ny=2, amplitude=0.01)
    vectors = []
    for dt in (0.04, 0.02, 0.01):
        result = evolve(model, duration=0.4, dt=dt).to_numpy()
        vectors.append(np.r_[result["displacement"].ravel(), result["velocity"].ravel()])
    coarse_error = np.linalg.norm(vectors[0] - vectors[1])
    fine_error = np.linalg.norm(vectors[1] - vectors[2])
    assert fine_error > 1e-12
    assert 3.7 < coarse_error / fine_error < 4.3


def test_standing_wave_returns_after_one_period():
    model, meta = standing_wave(device="cpu", nx=32, ny=4, amplitude=1e-4)
    state = evolve(model, duration=meta["period"], dt=meta["dt"])
    result = state.to_numpy()
    expected = meta["analytic"](result["points"][:, :2], result["time"])
    relative_height_error = np.sqrt(np.mean((result["surface_height"] - expected["surface_height"])**2)) / meta["amplitude"]
    assert relative_height_error < 0.003


def test_sloshing_translation_against_nonlinear_solution():
    model, meta = sloshing(device="cpu", rings=4, sectors=32)
    initial = model.state()
    reference_xy = initial.to_numpy()["points"][:, :2]
    state = evolve(model, duration=meta["period"] / 8, dt=meta["dt"])
    result = state.to_numpy()
    expected = meta["analytic"](reference_xy, result["time"])
    displacement_error = np.sqrt(np.mean(np.sum((result["displacement"] - expected["displacement"])**2, axis=1)))
    assert displacement_error / abs(meta["amplitude"]) < 0.01
    assert model.diagnostics(state)["min_j"] > 0.99
    assert relative_energy_change(model, initial, state) < 1e-6


def test_sloshing_shoreline_refinement_improves_translation():
    errors = []
    for rings, sectors in ((3, 16), (6, 32)):
        model, meta = sloshing(device="cpu", rings=rings, sectors=sectors)
        reference_xy = model.state().to_numpy()["points"][:, :2]
        state = evolve(model, duration=meta["period"] / 10, dt=meta["dt"])
        result = state.to_numpy()
        expected = meta["analytic"](reference_xy, result["time"])
        errors.append(np.sqrt(np.mean(np.sum((result["displacement"] - expected["displacement"])**2, axis=1))))
    # Combined spatial/shoreline refinement, with correspondingly smaller dt.
    # This is not claimed as a formal spatial-order estimate.
    assert errors[1] < 0.7 * errors[0]


@pytest.mark.slow
def test_sloshing_three_periods_remains_valid():
    model, meta = sloshing(device="cpu")
    initial = model.state()
    state = evolve(model, duration=3 * meta["period"], dt=meta["dt"])
    result = state.to_numpy()
    expected = meta["analytic"](model.points, result["time"])
    rms = np.sqrt(np.mean(np.sum((result["displacement"] - expected["displacement"])**2, axis=1)))
    assert rms / abs(meta["amplitude"]) < 0.08
    assert model.diagnostics(state)["min_j"] > 0.99
    assert relative_energy_change(model, initial, state) < 1e-6


@pytest.mark.parametrize("kwargs", [{"depth": 0}, {"amplitude": 1.0}, {"gravity": -1}])
def test_standing_wave_rejects_invalid_physics(kwargs):
    with pytest.raises(ValueError):
        standing_wave(device="cpu", **kwargs)
