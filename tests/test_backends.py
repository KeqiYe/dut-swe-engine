"""Numerical backend equivalence and rejected-step state preservation."""

import importlib.util

import numpy as np
import pytest

from swe import (
    DisplacementSolver,
    FlatBottom,
    GaussianBottom,
    GridBottom,
    Mesh,
    ModelBuilder,
    NumericalError,
    ParabolicBottom,
)


def _has_cuda():
    if importlib.util.find_spec("warp") is None:
        return False
    from swe.backends.warp_backend import available
    return available()


CUDA_AVAILABLE = _has_cuda()
CUDA = pytest.param(
    "cuda:0", marks=[pytest.mark.cuda, pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")]
)
DEVICES = ["cpu", "warp:cpu", CUDA]


def _require_device(device):
    if device.startswith("warp") or device.startswith("cuda"):
        pytest.importorskip("warp")


def _terrain(name):
    if name == "flat":
        return FlatBottom(z=-0.1)
    if name == "gaussian":
        return GaussianBottom(height=0.2, center=(0.7, 0.3), sigma=0.4, base=-0.1)
    if name == "parabolic":
        return ParabolicBottom(ax=0.1, ay=0.2, center=(0.4, 0.5), base=-0.1)
    return GridBottom.from_function(
        lambda xy: 0.04 * np.sin(2.0 * xy[:, 0]) * np.cos(xy[:, 1]),
        bounds=(-2.0, 3.0, -2.0, 3.0),
        shape=(17, 19),
    )


def _model(device, terrain="flat", manning=0.0, walls=True):
    _require_device(device)
    mesh = Mesh.rectangle(5, 4, size=(1.3, 0.9), walls=walls)
    builder = ModelBuilder(gravity=8.3, density=997.0, manning=manning)
    builder.add_water(
        mesh=mesh,
        bathymetry=_terrain(terrain),
        surface_height=lambda xy: 1.0 + 0.015 * np.cos(xy[:, 0]) * np.cos(2.0 * xy[:, 1]),
        velocity=lambda xy: np.column_stack((0.03 * np.sin(xy[:, 1]), 0.02 * np.cos(xy[:, 0]))),
    )
    return builder.finalize(device=device)


@pytest.mark.parametrize("device", ["warp:cpu", CUDA])
@pytest.mark.parametrize("terrain", ["flat", "gaussian", "parabolic", "grid"])
@pytest.mark.parametrize("manning", [0.0, 0.04])
def test_warp_matches_numpy_over_multiple_steps(device, terrain, manning):
    reference = _model("cpu", terrain, manning)
    actual = _model(device, terrain, manning)
    assert reference.fingerprint == actual.fingerprint
    # Nonzero deformation exercises the Jacobian and full cofactor terms.
    q = np.random.default_rng(81).normal(scale=2e-4, size=reference.points.shape)
    q[reference.fixed] = 0.0
    states = []
    for model in (reference, actual):
        state, scratch = model.state(), model.state()
        model.backend.copy(state.displacement, model.backend.array(q))
        states.append([state, scratch])
    solvers = [DisplacementSolver(reference), DisplacementSolver(actual)]
    for _ in range(12):
        for i, solver in enumerate(solvers):
            state, scratch = states[i]
            before_q = state.model.backend.numpy(state.displacement)
            before_v = state.model.backend.numpy(state.velocity)
            solver.step(state, scratch, dt=0.001)
            np.testing.assert_array_equal(state.model.backend.numpy(state.displacement), before_q)
            np.testing.assert_array_equal(state.model.backend.numpy(state.velocity), before_v)
            states[i] = [scratch, state]
    expected = states[0][0].to_numpy()
    result = states[1][0].to_numpy()
    for field in ("displacement", "velocity", "depth", "surface_height", "element_jacobian"):
        # Atomic addition orders differ across GPU schedules; bitwise equality
        # is not a scientific requirement for these double precision sums.
        np.testing.assert_allclose(result[field], expected[field], rtol=2e-10, atol=2e-12)
    assert result["time"] == expected["time"]
    assert result["step_index"] == expected["step_index"] == 12


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("failure", ["inverted", "singular", "nan_velocity", "inf_displacement", "candidate_inversion", "min_j"])
def test_rejected_step_preserves_input_and_output(device, failure):
    model = _model(device, walls=False)
    state, output = model.state(), model.state()
    backend = model.backend
    q = np.zeros_like(model.points)
    velocity = np.zeros_like(q)
    if failure == "inverted":
        q[:, 0] = -2.0 * model.points[:, 0]
    elif failure == "singular":
        q[:, 0] = -model.points[:, 0]
    elif failure == "nan_velocity":
        velocity[4, 1] = np.nan
    elif failure == "inf_displacement":
        q[4, 0] = np.inf
    elif failure == "candidate_inversion":
        velocity[:, 0] = -2000.0 * model.points[:, 0]
    backend.copy(state.displacement, backend.array(q))
    backend.copy(state.velocity, backend.array(velocity))
    out_q = np.full_like(q, 0.123)
    out_v = np.full_like(q, 0.456)
    backend.copy(output.displacement, backend.array(out_q))
    backend.copy(output.velocity, backend.array(out_v))
    output.time, output.step_index = 7.0, 17
    solver = DisplacementSolver(model, min_j=1.01 if failure == "min_j" else 1e-8)
    with pytest.raises(NumericalError):
        solver.step(state, output, dt=0.001)
    np.testing.assert_array_equal(backend.numpy(state.displacement), q)
    np.testing.assert_array_equal(backend.numpy(state.velocity), velocity)
    np.testing.assert_array_equal(backend.numpy(output.displacement), out_q)
    np.testing.assert_array_equal(backend.numpy(output.velocity), out_v)
    assert (state.time, state.step_index) == (0.0, 0)
    assert (output.time, output.step_index) == (7.0, 17)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_candidate", [False, True])
def test_raster_bounds_failure_is_transactional(device, at_candidate):
    model = _model(device, terrain="grid", walls=False)
    state, output = model.state(), model.state()
    backend = model.backend
    if at_candidate:
        backend.copy(state.velocity, backend.array(np.full_like(model.points, 5000.0)))
    else:
        backend.copy(state.displacement, backend.array(np.full_like(model.points, 5.0)))
    before_q = backend.numpy(state.displacement)
    before_v = backend.numpy(state.velocity)
    out_q = backend.numpy(output.displacement)
    out_v = backend.numpy(output.velocity)
    with pytest.raises(NumericalError, match="bathymetry"):
        DisplacementSolver(model).step(state, output, dt=0.001)
    np.testing.assert_array_equal(backend.numpy(state.displacement), before_q)
    np.testing.assert_array_equal(backend.numpy(state.velocity), before_v)
    np.testing.assert_array_equal(backend.numpy(output.displacement), out_q)
    np.testing.assert_array_equal(backend.numpy(output.velocity), out_v)
    assert (output.time, output.step_index) == (0.0, 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_candidate", [False, True])
def test_raster_rejects_vertices_outside_with_quadrature_inside(device, at_candidate):
    _require_device(device)
    from swe.backends.numpy_backend import geometry
    mesh = Mesh.rectangle(2, 2, size=(1.0, 1.0), walls=False)
    bottom = GridBottom.from_function(lambda xy: 0.0, (0.0, 1.0, 0.0, 1.0), shape=(3, 3))
    builder = ModelBuilder()
    builder.add_water(mesh=mesh, bathymetry=bottom, velocity=(0.1, 0.0))
    model = builder.finalize(device)
    state, output = model.state(), model.state()
    if not at_candidate:
        q = np.full_like(model.points, 1e-4)
        # This small translation puts vertices outside; all three interior
        # quadrature positions per triangle remain within the sampled domain.
        _, quadrature, *_ = geometry(model, q)
        bottom.evaluate(quadrature)
        model.backend.copy(state.displacement, model.backend.array(q))
    initial_output = output.to_numpy()
    with pytest.raises(NumericalError, match="bathymetry"):
        DisplacementSolver(model).step(state, output, dt=0.001)
    np.testing.assert_array_equal(model.backend.numpy(output.displacement), initial_output["displacement"])
    np.testing.assert_array_equal(model.backend.numpy(output.velocity), initial_output["velocity"])
    assert output.time == 0.0 and output.step_index == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", [0.0, -0.001, np.inf, np.nan])
def test_invalid_time_step_is_rejected_without_advancing(device, dt):
    model = _model(device)
    state, output = model.state(), model.state()
    before = output.to_numpy()
    with pytest.raises(ValueError, match="dt"):
        DisplacementSolver(model).step(state, output, dt)
    after = output.to_numpy()
    np.testing.assert_array_equal(after["displacement"], before["displacement"])
    np.testing.assert_array_equal(after["velocity"], before["velocity"])
    assert output.time == 0.0 and output.step_index == 0


@pytest.mark.parametrize("min_j", [0.0, -0.1, np.inf, np.nan])
def test_invalid_jacobian_threshold_is_rejected(min_j):
    with pytest.raises(ValueError, match="min_j"):
        DisplacementSolver(_model("cpu"), min_j=min_j)


def test_auto_uses_numpy_when_cuda_is_unavailable(monkeypatch):
    pytest.importorskip("warp")
    from swe.backends import warp_backend
    from swe.backends.numpy_backend import NumpyBackend
    monkeypatch.setattr(warp_backend, "available", lambda: False)
    model = _model("auto")
    assert isinstance(model.backend, NumpyBackend)
    assert "unavailable" in model.backend_reason


def test_explicit_unavailable_cuda_is_not_silently_replaced_by_cpu():
    pytest.importorskip("warp")
    if CUDA_AVAILABLE:
        pytest.skip("requires a host without CUDA")
    with pytest.raises(RuntimeError):
        _model("cuda:0")


def test_available_handles_device_initialization_failure(monkeypatch):
    pytest.importorskip("warp")
    from swe.backends import warp_backend
    def failed_initialization():
        raise RuntimeError("driver unavailable")
    monkeypatch.setattr(warp_backend.wp, "get_cuda_devices", failed_initialization)
    assert warp_backend.available() is False
