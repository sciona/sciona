"""Synthetic orchestration checks; these tests do not claim native training."""
from dataclasses import replace

import numpy as np
import pytest

from sciona import stacking_meta_operations as current
from sciona import otto_meta_stage as historical


def fixture():
    labels = np.tile(np.arange(9), 4)
    tree = np.column_stack((np.arange(36), labels, np.ones(36)))
    layout = dict(tree_training=tree, tree_query=tree[:9]+1000,
                  neural_training=np.column_stack((tree, np.full(36, 17))),
                  neural_query=np.column_stack((tree[:9]+2000, np.full(9, 19))))
    return layout, labels, np.arange(36) % 4, [f'synthetic-{i}' for i in range(36)]


def choices():
    return {family: [dict(choice=0), dict(choice=1)] for family in current.FAMILIES}


def install_oracles(monkeypatch, module, calls):
    def tune(x, labels, folds, identities, *, family, seed, candidates):
        calls.append(('select', family, x.copy(), labels.copy(), folds.copy(), list(identities), seed))
        assert candidates == choices()[family]
        return dict(selected_index=0 if family == 'lasagne_neural' else 1,
                    log_losses=[.2, .5] if family == 'lasagne_neural' else [.5, .2],
                    models_per_candidate=4*current.RUNS[family])

    def learner(family):
        def fit(x, labels, q, *, seed, controls):
            calls.append(('refit', family, x.copy(), labels.copy(), q.copy(), seed, controls.copy()))
            assert controls == dict(choice=0 if family == 'lasagne_neural' else 1)
            rng = np.random.default_rng(current.FAMILIES.index(family))
            bag = rng.uniform(.01, 1, (current.RUNS[family], len(q), 9))
            return bag/bag.sum(axis=2, keepdims=True)
        return fit, current.RUNS[family]

    monkeypatch.setattr(module, 'tune', tune)
    monkeypatch.setattr(module, '_learner', learner)


def test_composition_matches_source_and_preserves_all_fit_operands(monkeypatch):
    old_calls, new_calls = [], []
    install_oracles(monkeypatch, historical, old_calls)
    install_oracles(monkeypatch, current, new_calls)
    args = fixture()
    progress = []
    old = historical.fit(*args, seed=12, candidates=choices())
    new = current.fit_decomposed(*args, seed=12, candidates=choices(), progress=progress.append)
    assert new == old
    assert progress == list(current.FAMILIES)
    assert len(old_calls) == len(new_calls) == 6
    for old_call, new_call in zip(old_calls, new_calls):
        for old_value, new_value in zip(old_call, new_call):
            if isinstance(old_value, np.ndarray):
                np.testing.assert_array_equal(old_value, new_value)
            else:
                assert old_value == new_value
    assert sum(s['refit_models'] for s in new['selection'].values()) == 1100
    assert sum(s['tuning_model_fits'] for s in new['selection'].values()) == 8800


@pytest.mark.parametrize('bad', ['duplicate_id', 'missing_family', 'empty_candidates',
    'empty_training', 'query_rows', 'query_width', 'nan', 'folds', 'missing_class', 'seed'])
def test_all_input_checks_precede_training(monkeypatch, bad):
    layout, y, folds, ids = fixture()
    candidates, seed = choices(), 12
    if bad == 'duplicate_id': ids[0] = ids[1]
    elif bad == 'missing_family': del candidates['lasagne_neural']
    elif bad == 'empty_candidates': candidates['adaboost_extratrees'] = []
    elif bad == 'empty_training': layout['tree_training'] = np.zeros((0, 3))
    elif bad == 'query_rows': layout['neural_query'] = layout['neural_query'][:-1]
    elif bad == 'query_width': layout['tree_query'] = layout['tree_query'][:, :-1]
    elif bad == 'nan': layout['tree_query'][0, 0] = np.nan
    elif bad == 'folds': folds[0] = 4
    elif bad == 'missing_class': folds[y == 8] = 0
    else: seed = True
    def forbidden(*args, **kwargs): raise AssertionError('Training began')
    monkeypatch.setattr(current, 'tune', forbidden)
    with pytest.raises(ValueError):
        current.fit_decomposed(layout, y, folds, ids, seed=seed, candidates=candidates)


def test_detached_inputs_and_refit_selection_boundary(monkeypatch):
    install_oracles(monkeypatch, current, [])
    args, cs = fixture(), choices()
    prepared = current.prepare_meta_inputs(*args, 12, cs)
    selected = current.select_family(prepared, 'xgboost')
    cs['xgboost'][1]['choice'] = 19
    args[0]['tree_training'][0, 0] = 9999
    args[3][0] = 'changed'
    assert prepared.pairs['xgboost'][0][0, 0] == 0
    assert prepared.identities[0] == 'synthetic-0'
    assert selected.controls == {'choice': 1}
    assert 'synthetic' not in repr(prepared)
    assert current.refit_family(prepared, selected).shape == (250, 9, 9)
    other = current.prepare_meta_inputs(*fixture(), 12, choices())
    with pytest.raises(ValueError, match='these prepared inputs'):
        current.refit_family(other, selected)
    with pytest.raises(ValueError, match='controls changed'):
        current.refit_family(prepared, replace(selected, controls={'choice': 7}))


@pytest.mark.parametrize('bad', ['runs', 'shape', 'probabilities'])
def test_short_or_invalid_refit_bags_rejected(monkeypatch, bad):
    install_oracles(monkeypatch, current, [])
    prepared = current.prepare_meta_inputs(*fixture(), 12, choices())
    selected = current.select_family(prepared, 'xgboost')
    def learner(family):
        return (lambda *a, **k: np.full((249 if bad == 'shape' else 250, 9, 9),
                                       1. if bad == 'probabilities' else 1/9)), (249 if bad == 'runs' else 250)
    monkeypatch.setattr(current, '_learner', learner)
    with pytest.raises(ValueError): current.refit_family(prepared, selected)


def test_serialized_graph_runs_every_stage_with_source_result(monkeypatch, tmp_path):
    import asyncio
    from sciona.stacking_meta_graph import build_meta_graph
    from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
    from sciona.visualizer import runner

    calls = []
    install_oracles(monkeypatch, current, calls)
    args = fixture()
    expected = current.fit_decomposed(*args, seed=12, candidates=choices())
    calls.clear()
    digest, nodes, edges = encode_execution_graph(build_meta_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(graph)[0] == digest
    assert len(graph.nodes) == 12 and len(graph.edges) == 19
    captured = {}
    def capture(directory, node, name, value):
        captured[(node, name)] = value
    monkeypatch.setattr(runner, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(runner, 'save_intermediate_value', capture)
    layout, y, folds, ids = args
    result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-meta', 'case').execute(
        dict(layout=layout, labels=y, folds=folds, identities=ids, seed=12, candidates=choices()), cdg=graph))
    assert result['status'] == 'completed'
    assert captured[('result', 'out_result')] == expected
    assert {node for node, port in captured} == {node.node_id for node in graph.nodes}
    assert len(calls) == 6
