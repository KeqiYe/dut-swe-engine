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
