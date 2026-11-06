"""Preserve complete pseudo tails, exact arithmetic and private input integrity."""
import dataclasses
import numpy as np
import pytest
from sciona import decomposed_santander_runtime as runtime
from sciona.santander_blend import blend_neural_tree
from sciona.santander_execution import prepare
from sciona.decomposed_santander_graph import build_decomposed_santander_graph
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from scripts.santander_synthetic import payload


@pytest.fixture
def prepared():
    return prepare(payload())


@pytest.mark.parametrize('a,b', [([1,1,3,2],[4,2,2,1]), ([-3,0,8],[2,-5,9]), ([1],[9])])
def test_exact_original_blend_with_ties(a,b):
    fitted = {'neural': {'mean_ranks': a}, 'tree': {'mean_ranks': b}}
    values, coefficients = runtime.blend_inputs(fitted)
    assert np.array_equal(ordered_weighted_sum(values, coefficients)/3.1, blend_neural_tree(a,b))
    assert np.array_equal(coefficients, [2.1,1.])
    assert fitted['neural']['mean_ranks'] == a


@pytest.mark.parametrize('a,b', [([],[]), ([1],[1,2]), ([np.nan],[1]), ([1],[np.inf]), ([[1]],[[2]]), (['a'],[1])])
def test_invalid_blend_rejected(a,b):
    with pytest.raises(ValueError):
        runtime.blend_inputs({'neural': {'mean_ranks': a}, 'tree': {'mean_ranks': b}})


def test_full_distinct_selection_tails(prepared):
    selected = runtime.selections(prepared, {'mean_ranks': np.arange(8500.)})
    for name, positive, negative, total in [('neural',5000,3000,8060),('tree',2700,2000,4760)]:
        v = selected[name]
        assert np.sum(v['selected_labels']==1) == positive
        assert np.sum(v['selected_labels']==0) == negative
        assert len(v['labels']) == total and len(v['query']) == 8500
        assert len(set(v['selected_positions'])) == positive+negative


def test_ambiguous_selection_rejected(prepared):
    with pytest.raises(ValueError):
        runtime.selections(prepared, {'mean_ranks': np.ones(8500)})


def test_modified_preparation_rejected(prepared):
    with pytest.raises(ValueError):
        runtime.initial(dataclasses.replace(prepared, fingerprint='0'*64))


def test_complete_training_functions_preserved():
    from sciona import santander_lifecycle, santander_retraining
    assert runtime.train_neural_cv is santander_lifecycle.train_neural_cv
    assert runtime.train_tree_cv is santander_retraining.train_tree_cv
    assert runtime.expanded_population is santander_retraining.expanded_population


def test_all_stages_and_generic_boundary_present():
    graph = build_decomposed_santander_graph()
    assert len(graph.nodes)==7 and len(graph.edges)==13
    generic = next(n for n in graph.nodes if n.node_id=='blend')
    assert generic.matched_primitive.endswith('ordered_sum.ordered_weighted_sum')
    assert {(e.source_id,e.input_name) for e in graph.edges if e.target_id=='result'} == {
        ('prepare','prepared'),('initial','initial_models'),('selections','selected'),
        ('branches','fitted'),('blend','weighted')}


def test_result_preserves_counts_and_query_order(prepared):
    selected = runtime.selections(prepared, {'mean_ranks': np.arange(8500.)})
    fitted = {name: {'models': list(range(10))} for name in ('neural','tree')}
    weighted = np.arange(8500.)
    result = runtime.result(prepared, {'models': list(range(10))}, selected, fitted, weighted)
    assert np.array_equal(result['scores'], weighted/3.1)
    assert result['model_counts']==dict(initial_neural=10,neural=10,tree=10)
    with pytest.raises(ValueError):
        runtime.result(prepared, {'models': []}, selected, fitted, weighted[:-1])
