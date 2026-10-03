"""Topology and physics checks for the public-package refined island example."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).parents[1] / "examples" / "island_refined.py"
    spec = importlib.util.spec_from_file_location("island_refined_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tenfold_island_is_conforming_wet_and_constrained(example):
    model, settings = example.build_model(device="cpu", mesh_factor=10)
    assert len(model.triangles) == 10060
    assert len(model.points) == 5118
    edges = np.sort(np.concatenate((model.triangles[:, [0, 1]],
                                    model.triangles[:, [1, 2]],
                                    model.triangles[:, [2, 0]])), axis=1)
    edges, count = np.unique(edges, axis=0, return_counts=True)
    assert np.all((count == 1) | (count == 2))
    # A connected domain with one island hole has Euler characteristic zero.
    assert len(model.points) - len(edges) + len(model.triangles) == 0
    coordinates = {tuple(xy) for xy in np.round(model.points, 11)}
    midpoints = np.round(model.points[edges].mean(axis=1), 11)
    assert not any(tuple(xy) in coordinates for xy in midpoints)
    # Interior bisection must leave the projected boundary and its wall DOFs.
    boundary = np.unique(edges[count == 1])
    coarse, _ = example.build_model(device="cpu", mesh_factor=4)
    np.testing.assert_array_equal(model.points[boundary], coarse.points[boundary])
    np.testing.assert_array_equal(model.fixed[:len(coarse.points)], coarse.fixed)
    assert not model.fixed[len(coarse.points):].any()
    assert model.depth0.min() > 0
    assert model.node_depth0.min() >= -1e-12
    lengths = np.linalg.norm(model.points[model.triangles] -
                             np.roll(model.points[model.triangles], 1, axis=1), axis=-1)
    assert np.max(lengths.max(axis=1)**2 / (2 * model.area)) < 7
    assert settings["dt"] < model.suggested_dt()
    import swe
    state = swe.run(model, dt=settings["dt"], duration=0.025)
    assert model.diagnostics(state)["min_j"] > 0.99


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_small_integer_mesh_factors_are_supported(example, factor):
    model, settings = example.build_model(device="cpu", mesh_factor=factor)
    assert len(model.triangles) == factor * 1006
    assert settings["mesh_factor"] == factor
    assert model.depth0.min() > 0


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_mesh_factor_requires_positive_integer(example, factor):
    with pytest.raises(ValueError, match="positive integer"):
        example.build_model(device="cpu", mesh_factor=factor)


def test_default_island_preserves_reference_initial_state(example):
    from swe.cases import island

    model, settings = example.build_model(device="cpu")
    reference, reference_settings = island(device="cpu")
    assert model.fingerprint == reference.fingerprint
    assert settings["dt"] == reference_settings["dt"]
    assert settings["amplitude"] == 0.032


@pytest.mark.parametrize("factor", [1, 10])
def test_taller_wave_changes_physics_and_preserves_wet_coast(example, factor):
    model, settings = example.build_model(device="cpu", mesh_factor=factor, amplitude=0.064)
    reference, original_settings = example.build_model(device="cpu", mesh_factor=factor)
    crest = model.surface0.max() - settings["mean_height"]
    assert 0.062 < crest <= 0.064
    assert crest > 1.9 * (reference.surface0.max() - 0.32)
    assert model.initial_velocity[:, 0].max() > 1.8 * reference.initial_velocity[:, 0].max()
    assert model.depth0.min() > 0
    assert model.node_depth0.min() >= -1e-12
    edges, count = np.unique(example._edges(model.triangles), axis=0, return_counts=True)
    boundary = np.unique(edges[count == 1])
    shoreline = boundary[~model.fixed[boundary].any(axis=1)]
    np.testing.assert_allclose(model.node_depth0[shoreline], 0, atol=1e-12)
    assert settings["dt"] < original_settings["dt"]


@pytest.mark.parametrize("amplitude", [0, -0.01, np.nan, np.inf])
def test_amplitude_requires_finite_positive_value(example, amplitude):
    with pytest.raises(ValueError, match="amplitude must be finite and positive"):
        example.build_model(device="cpu", amplitude=amplitude)


def test_pulse_broadens_without_moving_or_slowing_the_crest(example, monkeypatch):
    amplitude, mean_height = 0.064, 0.32
    base_k = np.sqrt(3 * amplitude / (4 * mean_height**2 * (mean_height + amplitude)))
    center = 5 / base_k
    # Sample the actual scene callbacks at the crest and on either side,
    # independently of whether a mesh node lands exactly on the crest.
    xy = np.column_stack(([center - 1, center, center + 1], [10., 10., 10.]))
    fields = []
    original_builder = example.swe.ModelBuilder

    class SamplingBuilder(original_builder):
        def add_water(self, **kwargs):
            fields.append((kwargs["surface_height"](xy), kwargs["velocity"](xy)))
            return super().add_water(**kwargs)

    monkeypatch.setattr(example.swe, "ModelBuilder", SamplingBuilder)
    _, narrow_settings = example.build_model(device="cpu", amplitude=amplitude)
    _, broad_settings = example.build_model(device="cpu", amplitude=amplitude, width_scale=2)
    narrow_eta, narrow_velocity = fields[0]
    broad_eta, broad_velocity = fields[1]
    assert broad_settings["wave_center"] == narrow_settings["wave_center"]
    assert broad_settings["width_scale"] == 2
    assert broad_eta[1] == narrow_eta[1] == mean_height + amplitude
    assert broad_velocity[1, 0] == narrow_velocity[1, 0]
    np.testing.assert_allclose(broad_eta[0], broad_eta[2], atol=1e-15)
    assert np.all(broad_eta[[0, 2]] > narrow_eta[[0, 2]])


@pytest.mark.parametrize("factor", [1, 10])
def test_width_only_change_reprojects_wet_shore(example, factor):
    model, _ = example.build_model(device="cpu", mesh_factor=factor, width_scale=2)
    assert model.depth0.min() > 0
    assert model.node_depth0.min() >= -1e-12
    edges, count = np.unique(example._edges(model.triangles), axis=0, return_counts=True)
    boundary = np.unique(edges[count == 1])
    shoreline = boundary[~model.fixed[boundary].any(axis=1)]
    np.testing.assert_allclose(model.node_depth0[shoreline], 0, atol=1e-12)


@pytest.mark.parametrize("width_scale", [0, -1, np.nan, np.inf])
def test_width_scale_requires_finite_positive_value(example, width_scale):
    with pytest.raises(ValueError, match="width_scale must be finite and positive"):
        example.build_model(device="cpu", width_scale=width_scale)
