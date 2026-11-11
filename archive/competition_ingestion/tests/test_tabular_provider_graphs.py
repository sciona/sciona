"""Uncached serialized numeric provider calls on synthetic inputs."""
import numpy as np
import pytest
from sciona.tabular_provider_graphs import PROVIDERS, build_provider_graph
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('name', ['frequency_fit', 'frequency_apply', 'target', 'normalize', 'indices'])
def test_serialized_numeric_provider(name, tmp_path, monkeypatch):
    inputs = dict(
        frequency_fit=dict(categories=np.array(['b', 'a', 'b'], dtype=object)),
        frequency_apply=dict(categories=np.array(['new', 'b', 'b'], dtype=object), frequency_map={'b': .75}, unknown_value=2.),
        target=dict(categories=np.array(['b', 'a', 'b'], dtype=object), targets=np.array([-10., 40., 20.]), smoothing=2., prior=-4.),
        normalize=dict(doc_topic_distr=np.array([[2., 0., 6.], [1., 2., 1.]])),
        indices=dict(support_mask=np.array([False, True, False, True])),
    )[name]
    expected = PROVIDERS[name](**inputs)
    actual = run(build_provider_graph(name), inputs, tmp_path, monkeypatch)[name, 'out_result']
    if isinstance(expected, np.ndarray): np.testing.assert_array_equal(actual, expected)
    else: assert actual == expected


@pytest.mark.parametrize('stat', ['count', 'sum', 'mean', 'var', 'std', 'min', 'max', 'first', 'last'])
def test_serialized_group_reduction_branches(stat, tmp_path, monkeypatch):
    inputs = dict(values=np.array([2., -4., 2., 10.]), groups=np.array(['b', 'a', 'b', 'b'], dtype=object), agg_fn=stat)
    expected = PROVIDERS['aggregate'](**inputs)
    actual = run(build_provider_graph('aggregate'), inputs, tmp_path, monkeypatch)['aggregate', 'out_result']
    np.testing.assert_array_equal(actual, expected)


def test_native_fitted_frequency_map_can_be_reused_for_new_query(tmp_path, monkeypatch):
    fitted = run(build_provider_graph('frequency_fit'),
        dict(categories=np.array(['a', 'b', 'a', 'a'], dtype=object)), tmp_path, monkeypatch)['frequency_fit', 'out_result']
    before = dict(fitted)
    actual = run(build_provider_graph('frequency_apply'), dict(categories=np.array(['b', 'new', 'a', 'a'], dtype=object),
        frequency_map=fitted, unknown_value=0.), tmp_path, monkeypatch)['frequency_apply', 'out_result']
    np.testing.assert_array_equal(actual, [.25, 0., .75, .75])
    assert fitted == before
