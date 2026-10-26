"""All-entry dispatch parity plus real shared-neighbor execution, synthetic only."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from sciona import otto_first_level as historical
from sciona import stacking_first_operations as current
from sciona import otto_knn_models as knn
from sciona.otto_tsne import PopulationEmbedding


def fixture():
    rng = np.random.default_rng(74)
    y = np.arange(1350) % 9
    x = np.column_stack((np.eye(9)[y]*10, np.ones((1350, 4))))
    x += rng.uniform(0, .2, x.shape)
    q = x[:9].copy()
    ids = [f'synthetic-train-{i}' for i in range(1350)]
    qids = [f'synthetic-query-{i}' for i in range(9)]
    embedding = PopulationEmbedding(tuple(ids+qids), np.zeros((1359, 3)), 0.)
    controls = {i: {'tag': i} for i in range(1, 24)}
    controls.update({i: {'metric': 'euclidean'} for i in range(24, 34)})
    controls[25] = controls[30] = {'metric': 'cityblock'}
    return (x, y, np.arange(1350) % 5, ids, q, qids), embedding, controls


def assert_same(a, b):
    if isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, PopulationEmbedding):
        assert a.identities == b.identities
        np.testing.assert_array_equal(a.coordinates, b.coordinates)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_same(x, y)
    else: assert a == b


def test_all_entries_preserve_dispatch_operands_and_shared_bank_counts(monkeypatch):
    args, embedding, controls = fixture()
    original_import = importlib.import_module
    calls = []

    def produce(name, function, *values, **kwargs):
        calls.append((name, function, values, kwargs))
        columns = (10, 9) if name == 'knn_models' else (9,)
        return dict(oof=np.full((len(values[0]), *columns), len(calls)/100),
                    query=np.full((len(values[4]), *columns), len(calls)/100))

    def load(name, *args, **kwargs):
        if not name.startswith('sciona.otto_'):
            return original_import(name, *args, **kwargs)
        family = name.removeprefix('sciona.otto_')
        class Module:
            def __getattr__(self, function):
                return lambda *a, **k: produce(family, function, *a, **k)
        return Module()

    def clusters(embedding, identities, **kwargs):
        calls.append(('embedding_clusters', 'fit', (embedding, identities), kwargs))
        return SimpleNamespace(lookup=lambda ids: np.full((len(ids), 5), .1))

    monkeypatch.setattr(importlib, 'import_module', load)
    monkeypatch.setattr(historical, 'fit_embedding_clusters', clusters)
    monkeypatch.setattr(current, 'fit_embedding_clusters', clusters)
    monkeypatch.setattr(knn, 'crossfit', lambda *a, **k: produce('knn_models', 'crossfit', *a, **k))
    old_progress, new_progress = [], []
    old = historical.build(*args, embedding=embedding, seed=12, controls=controls, progress=old_progress.append)
    old_calls = list(calls)
    calls.clear()
    new = current.build_decomposed(*args, embedding=embedding, seed=12, controls=controls, progress=new_progress.append)
    assert old_progress == new_progress == list(range(1, 34))
    assert_same(old_calls, calls)
    assert len([c for c in calls if c[0] == 'knn_models']) == 2
    assert len(calls) == 25  # 23 direct entries, two shared neighbor banks.
    for entry in range(1, 34):
        for field in ('training', 'query', 'training_ids', 'query_ids', 'folds'):
            assert_same(getattr(old[entry], field), getattr(new[entry], field))
    assert current.neighbor_groups(controls) == [[24, 26, 27, 28, 29, 31, 32, 33], [25, 30]]


def test_real_neighbor_bank_matches_direct_native_result_and_preserves_alignment():
    args, embedding, controls = fixture()
    prepared = current.prepare_first_inputs(*args, embedding, 12, controls)
    bank = current.fit_neighbor_bank(prepared, 24)
    expected = knn.crossfit(*args, metric='euclidean')
    for entry in current.neighbor_groups(controls)[0]:
        block = current.select_neighbor_entry(prepared, bank, entry)
        np.testing.assert_array_equal(block.training, expected['oof'][:, entry-24])
        np.testing.assert_array_equal(block.query, expected['query'][:, entry-24])
        np.testing.assert_array_equal(block.folds, args[2])
        assert block.training_ids == tuple(args[3]) and block.query_ids == tuple(args[5])
    with pytest.raises(ValueError, match='control group'):
        current.select_neighbor_entry(prepared, bank, 25)
    other = current.prepare_first_inputs(*args, embedding, 12, controls)
    with pytest.raises(ValueError, match='these prepared inputs'):
        current.select_neighbor_entry(other, bank, 24)


@pytest.mark.parametrize('bad', ['controls', 'overlap', 'references', 'folds', 'class_partition', 'seed'])
def test_invalid_population_rejected_before_producers(monkeypatch, bad):
    args, embedding, controls = fixture()
    args = list(args)
    seed = 12
    if bad == 'controls': del controls[33]
    elif bad == 'overlap': args[5][0] = args[3][0]
    elif bad == 'references':
        args[:4] = [v[:1200] for v in args[:4]]
    elif bad == 'folds': args[2][0] = 5
    elif bad == 'class_partition': args[2][args[1] == 8] = 0
    else: seed = True
    def forbidden(*a, **k): raise AssertionError('Producer started')
    monkeypatch.setattr(current, 'fit_entry', forbidden)
    with pytest.raises(ValueError):
        current.build_decomposed(*args, embedding=embedding, seed=seed, controls=controls)


def test_detached_context_and_entry_ranges():
    args, embedding, controls = fixture()
    prepared = current.prepare_first_inputs(*args, embedding, 12, controls)
    original = prepared.args[0][0, 0]
    args[0][0, 0] = 999
    controls[1]['tag'] = 777
    embedding.coordinates[0, 0] = 555
    assert prepared.args[0][0, 0] == original
    assert prepared.controls[1] == {'tag': 1}
    assert prepared.embedding.coordinates[0, 0] == 0
    for entry in (0, 24, True):
        with pytest.raises(ValueError): current.fit_entry(prepared, entry)
    for entry in (23, 34, True):
        with pytest.raises(ValueError): current.fit_neighbor_bank(prepared, entry)


def test_graph_executes_all_entries_and_two_shared_groups(monkeypatch, tmp_path):
    import asyncio
    import json
    from sciona.stacking_first_graph import build_first_graph
    from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
    from sciona.visualizer import runner

    args, embedding, controls = fixture()
    calls, captured = [], {}
    def entry(inputs, index):
        calls.append(('entry', index))
        width = 5 if index == 10 else 9
        return current._block(inputs, index, np.full((1350, width), index/100), np.full((9, width), index/100))
    def bank(inputs, representative):
        calls.append(('bank', representative))
        raw = {key: np.broadcast_to(np.arange(24,34)[None,:,None]/100, (size,10,9)).copy()
               for key, size in (('oof',1350),('query',9))}
        return current.NeighborBank(inputs, json.dumps(inputs.controls[representative],sort_keys=True),raw)
    monkeypatch.setattr(current,'fit_entry',entry)
    monkeypatch.setattr(current,'fit_neighbor_bank',bank)
    graph = build_first_graph(current.neighbor_groups(controls))
    digest,nodes,edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0] == digest
    assert len(graph.nodes) == 38
    def capture(directory,node,port,value):
        if port.startswith('out_'):captured[(node,port)] = value
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    monkeypatch.setattr(runner,'save_intermediate_value',capture)
    user_inputs = dict(zip(('training','labels','folds','training_ids','query','query_ids'),args))
    user_inputs.update(embedding=embedding,seed=12,controls=controls)
    result = asyncio.run(runner.CDGExecutionSession(None,'synthetic-first','case').execute(user_inputs,cdg=graph))
    assert result['status'] == 'completed'
    assert {n for n,p in captured} == {n.node_id for n in graph.nodes}
    assert sorted(i for kind,i in calls if kind=='entry') == list(range(1,24))
    assert sorted(i for kind,i in calls if kind=='bank') == [24,25]
    blocks=captured[('collect','out_first_level')]
    for index in range(1,34):
        np.testing.assert_array_equal(blocks[index].training,index/100)
        np.testing.assert_array_equal(blocks[index].query,index/100)


@pytest.mark.parametrize('groups', [[],[[24]], [list(range(24,34)),[24]],[[True]+list(range(25,34))]])
def test_graph_rejects_incomplete_or_duplicate_neighbor_partition(groups):
    from sciona.stacking_first_graph import build_first_graph
    with pytest.raises(ValueError):build_first_graph(groups)


@pytest.mark.parametrize('bad', ['missing', 'identities', 'folds', 'missing_folds',
                                'embedding_folds', 'shape', 'nonfinite'])
def test_collector_rejects_incomplete_or_misaligned_blocks(bad):
    from dataclasses import replace
    args, embedding, controls = fixture()
    inputs = current.prepare_first_inputs(*args, embedding, 12, controls)
    blocks = {i: current._block(inputs, i, np.full((1350, 5 if i == 10 else 9), .1),
                               np.full((9, 5 if i == 10 else 9), .1)) for i in range(1, 34)}
    if bad == 'missing': del blocks[33]
    elif bad == 'identities': blocks[2] = replace(blocks[2], training_ids=blocks[2].training_ids[::-1])
    elif bad == 'folds': blocks[2] = replace(blocks[2], folds=np.roll(blocks[2].folds, 1))
    elif bad == 'missing_folds': blocks[1] = replace(blocks[1], folds=None)
    elif bad == 'embedding_folds': blocks[10] = replace(blocks[10], folds=blocks[1].folds)
    elif bad == 'shape': blocks[2] = replace(blocks[2], query=blocks[2].query[:-1])
    else: blocks[2].training[0, 0] = np.nan
    with pytest.raises(ValueError): current.collect_first_blocks(blocks)
