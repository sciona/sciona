"""Uncached native numeric execution checked against independent oracles."""
import numpy as np
import pytest
from sciona.constrained_array_graphs import build_provider_graph, build_composed_graph
from tests.test_constrained_array_ops import exhaustive_projection
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('totals', [[3., 0.], [1., 7.]])
def test_native_projection(totals, tmp_path, monkeypatch):
    values = np.array([[-1., 2., 3.], [4., -2., .5]])
    before = values.copy()
    totals = np.asarray(totals)
    actual = run(build_provider_graph('project'), dict(values=values, totals=totals), tmp_path, monkeypatch)['project', 'out_result']
    expected = np.stack([exhaustive_projection(row, total) for row, total in zip(values, totals)])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize('weight', [0., .25, .5])
def test_native_signed_smoothing(weight, tmp_path, monkeypatch):
    values = np.array([[-2., 4., 1.], [3., -1., 2.]])
    matrix = np.array([[1-2*weight, weight, weight], [weight, 1-2*weight, weight], [weight, weight, 1-2*weight]])
    actual = run(build_provider_graph('smooth'), dict(values=values, weight=weight), tmp_path, monkeypatch)['smooth', 'out_result']
    np.testing.assert_allclose(actual, values @ matrix.T, atol=1e-12)
    np.testing.assert_allclose(actual.sum(1), values.sum(1), atol=1e-12)


def test_native_composition_routes_array(tmp_path, monkeypatch):
    captured = run(build_composed_graph(), dict(values=np.array([[-1., 2., 3.]]),
        totals=np.array([3.]), weight=.25), tmp_path, monkeypatch)
    assert captured['project', 'out_result'] is captured['smooth', 'in_values']
    np.testing.assert_allclose(captured['project', 'out_result'], [[0., 1., 2.]])
    np.testing.assert_allclose(captured['smooth', 'out_result'], [[.75, 1., 1.25]])
