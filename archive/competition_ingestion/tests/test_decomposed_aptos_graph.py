"""Structural and boundary checks; these tests do not qualify model training."""
import ast
import copy
import inspect

import numpy as np
import pytest

from sciona import aptos_pipeline as reference
from sciona import decomposed_aptos_pipeline as candidate
from sciona.aptos_ensemble import classify
from sciona.atoms.numerical.statistics.cut_indices import ordered_cut_indices
from sciona.decomposed_aptos_graph import build_decomposed_aptos_graph
from scripts.build_aptos_execution_graph import build_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph


def test_complete_training_body_is_unchanged():
    original = ast.parse(inspect.getsource(reference.run_reference)).body[0]
    actual = ast.parse(inspect.getsource(candidate.train_scores)).body[0]
    original.name = actual.name
    original.body.pop(0)
    actual.body.pop(0)
    context = next(n for n in original.body if isinstance(n, ast.With))
    last = context.body.pop()
    assert isinstance(last, ast.Assign) and ast.unparse(last) == 'ordinals = classify(scores, tie_policy=tie_policy)'
    returned = original.body[-1].value
    position = next(i for i, key in enumerate(returned.keys) if key.value == 'ordinals')
    returned.keys[position] = ast.Constant('tie_policy')
    returned.values[position] = ast.Name(id='tie_policy', ctx=ast.Load())
    assert ast.dump(original, include_attributes=False) == ast.dump(actual, include_attributes=False)
    for name in ('_predict', '_batches', '_digest', '_restore', 'build_from_checkpoint',
                 'prepare_population', 'first_stage_targets', 'second_stage_targets', 'train_epochs', 'blend'):
        assert getattr(candidate, name) is getattr(reference, name)


def test_graph_preserves_root_contract_and_codec():
    graph = build_decomposed_aptos_graph()
    original = build_graph()
    assert len(graph.nodes) == 5 and len(graph.edges) == 7
    assert graph.nodes[0] == original.nodes[0]
    assert graph.nodes[1].inputs == original.nodes[1].inputs
    assert {n.node_id for n in graph.nodes} == {'population', 'train_scores', 'cut_inputs', 'classify', 'result'}
    assert graph.metadata['catalog_output_sources'] == {'result': {'node_id': 'result', 'output_name': 'result'}}
    digest, nodes, edges = encode_execution_graph(graph)
    assert decode_execution_graph(nodes, edges, digest) == graph
    connected = {(e.target_id, e.input_name) for e in graph.edges}
    roots = [(n.node_id, p.name) for n in graph.nodes for p in n.inputs if (n.node_id, p.name) not in connected]
    expected = [('population', p.name) for p in original.nodes[0].inputs]
    expected += [('train_scores', p.name) for p in original.nodes[1].inputs if p.name != 'plan']
    assert roots == expected


def training_result(policy='upper'):
    # Deliberately not a trained result: only encoding and graph-boundary contracts are tested here.
    return dict(scores=np.array([-.2, .7, 1.5, 2.5, 3.5, 9.]), tie_policy=policy,
        fits=[], first_stage_predictions={}, second_stage_predictions={}, second_stage_targets=np.array([]),
        checkpoint_replays_exact=True, temporary_checkpoints_removed=True)


@pytest.mark.parametrize('policy', ['upper', 'lower'])
def test_recipe_policy_generic_operation_and_result(policy):
    trained = training_result(policy)
    before = copy.deepcopy(trained)
    values, cuts, ties = candidate.cut_inputs(trained)
    classes = ordered_cut_indices(values, cuts, ties)
    result = candidate.result(trained, classes)
    np.testing.assert_array_equal(result['ordinals'], classify(trained['scores'], tie_policy=policy))
    assert set(result) == set(trained)-{'tie_policy'} | {'ordinals'}
    assert result['fits'] is trained['fits']
    np.testing.assert_array_equal(trained['scores'], before['scores'])
    assert not np.shares_memory(values, trained['scores'])
    assert not np.shares_memory(result['ordinals'], classes)


@pytest.mark.parametrize('scores', [np.array([]), np.array([np.nan]), np.array([np.inf]),
    np.array([[1.]]), np.array([1], dtype=np.int64), np.array([1.], dtype=np.float32)])
def test_invalid_recipe_scores_reject(scores):
    trained = training_result()
    trained['scores'] = scores
    with pytest.raises(ValueError):
        candidate.cut_inputs(trained)


@pytest.mark.parametrize('ordinals', [np.array([0]), np.zeros(6), np.full(6, -1), np.full(6, 5)])
def test_invalid_result_classes_reject(ordinals):
    with pytest.raises(ValueError):
        candidate.result(training_result(), ordinals)


def test_missing_result_fields_reject():
    trained = training_result()
    del trained['fits']
    with pytest.raises(ValueError):
        candidate.result(trained, np.zeros(6, dtype=np.int64))


def test_unknown_recipe_tie_policy_rejects():
    with pytest.raises(ValueError):
        candidate.cut_inputs(training_result('nearest'))
