"""Check full recipe boundaries and exact reduction order on synthetic values."""
import dataclasses
import numpy as np
import pytest
from sciona import decomposed_porto_runtime as runtime, porto_ensemble
from sciona.porto_execution import prepare
from sciona.porto_tree_blend import blend
from sciona.decomposed_porto_graph import build_decomposed_porto_graph
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from scripts.porto_synthetic import payload


@pytest.mark.parametrize('width',[1,8,17,1000])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_exact_probability_reduction(width,dtype):
    rng=np.random.default_rng(52)
    neural=rng.uniform(size=(5,width)).astype(dtype)
    tree=rng.uniform(size=width).astype(dtype)
    before=neural.copy()
    values,coefficients=runtime.blend_inputs(dict(neural_probabilities=neural,tree_probabilities=tree))
    assert np.array_equal((ordered_weighted_sum(values,coefficients)+tree)/6,blend(neural,tree))
    assert np.array_equal(neural,before)


@pytest.mark.parametrize('fault',['missing_model','misaligned_tree','nan','outside_probability'])
def test_invalid_probabilities(fault):
    neural=np.ones((5,8))*.5; tree=np.ones(8)*.5
    if fault=='missing_model':neural=neural[:-1]
    elif fault=='misaligned_tree':tree=tree[:-1]
    elif fault=='nan':neural[0,0]=np.nan
    else:tree[0]=1.01
    with pytest.raises(ValueError):runtime.blend_inputs(dict(neural_probabilities=neural,tree_probabilities=tree))


def test_prepared_integrity():
    prepared=prepare(payload())
    with pytest.raises(ValueError):runtime.features(dataclasses.replace(prepared,fingerprint='0'*64))


def test_feature_preparation_and_learner_identity():
    p=payload();prepared=prepare(p)
    expected=porto_ensemble.prepare_populations(p['training']['values'],p['query']['values'],**p['controls']['preparation'])
    actual=runtime.features(prepared)
    for key in expected:assert np.array_equal(actual[key],expected[key])
    assert runtime.fit_branch is porto_ensemble.fit_branch
    assert runtime.fit_tree is porto_ensemble.fit_tree
    assert runtime.prepare_populations is porto_ensemble.prepare_populations


def test_complete_graph():
    graph=build_decomposed_porto_graph()
    assert len(graph.nodes)==6 and len(graph.edges)==9
    assert next(n for n in graph.nodes if n.node_id=='blend').matched_primitive.endswith('ordered_sum.ordered_weighted_sum')
