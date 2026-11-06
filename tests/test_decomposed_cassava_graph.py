"""Structural and synthetic adapter checks, not full training qualification."""
import ast
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from sciona.atoms.ml import decomposed_cassava_execution as adapters
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from sciona.cassava_family_alignment import assemble
from sciona.decomposed_cassava_graph import build_decomposed_cassava_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from scripts.build_cassava_execution_graph import build_graph


def test_complete_training_body_preserved():
    root = Path(__file__).resolve().parents[1] / 'sciona'
    old = ast.parse((root / 'cassava_pipeline.py').read_text())
    new = ast.parse((root / 'decomposed_cassava_pipeline.py').read_text())
    old.body.pop(0)
    new.body.pop(0)
    removed = [n for n in old.body if isinstance(n, ast.ImportFrom)
               and n.module == 'sciona.cassava_family_alignment']
    assert len(removed) == 1
    old.body.remove(removed[0])
    original, actual = old.body[-1], new.body[-1]
    assert original.name == 'execute' and actual.name == 'train_families'
    original.name = actual.name
    original.body.pop(0)
    actual.body.pop(0)
    assert ast.unparse(original.body[-2]) == 'result = assemble(keys, **outputs)'
    assert ast.unparse(original.body[-1]) == 'return (result, outputs, histories, evaluations, artifacts)'
    original.body[-2:] = ast.parse('return dict(query_keys=keys, family_outputs=outputs, histories=histories, evaluations=evaluations, checkpoints=artifacts)').body
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_graph_contract_codec_and_registered_signatures():
    graph, original = build_decomposed_cassava_graph(), build_graph()
    assert (len(graph.nodes), len(graph.edges)) == (7, 11)
    assert graph.nodes[0] == original.nodes[0]
    assert graph.nodes[1].inputs == original.nodes[1].inputs
    providers = [n.matched_primitive for n in graph.nodes]
    assert len(set(providers)) == 6
    assert providers[3] == providers[5] == ordered_weighted_sum.__module__ + '.ordered_weighted_sum'
    for node in graph.nodes:
        module, name = node.matched_primitive.rsplit('.', 1)
        function = getattr(importlib.import_module(module), name)
        assert list(inspect.signature(function).parameters) == [p.name for p in node.inputs]
    connected = {(e.target_id, e.input_name) for e in graph.edges}
    roots = [(n.node_id, p.name) for n in graph.nodes for p in n.inputs
             if (n.node_id, p.name) not in connected]
    expected = [('fold_plan', p.name) for p in original.nodes[0].inputs]
    expected += [('train_families', p.name) for p in original.nodes[1].inputs if p.name != 'plan']
    assert roots == expected and len(roots) == 11
    for key in ('source_version_ids', 'choices', 'limitations', 'scope'):
        assert graph.metadata[key] == original.metadata[key]
    digest, nodes, edges = encode_execution_graph(graph)
    assert decode_execution_graph(nodes, edges, digest) == graph


def fixture():
    # Empty histories deliberately distinguish this from trained evidence.
    rng = np.random.default_rng(93)
    keys = ['synthetic-a', 'synthetic-b', 'synthetic-c']
    families = {name: (keys[::-1], rng.dirichlet(np.ones(5), size=3))
                for name in ('vit', 'resnext', 'efficientnet', 'cropnet')}
    return dict(query_keys=keys, family_outputs=families, histories={}, evaluations={}, checkpoints={})


def test_adapter_composition_preserves_source_result_and_evidence():
    trained = fixture()
    context, values, coefficients = adapters.pair_inputs(trained)
    paired = ordered_weighted_sum(values, coefficients)
    values, coefficients = adapters.final_inputs(context, paired)
    scores = ordered_weighted_sum(values, coefficients)
    actual = adapters.result(trained, context, scores)
    expected = assemble(trained['query_keys'], **trained['family_outputs'])
    assert set(actual) == {'predictions', 'family_outputs', 'histories', 'evaluations', 'checkpoints'}
    for field in ('scores', 'labels'):
        np.testing.assert_array_equal(actual['predictions'][field], expected[field])
    for field in ('family_outputs', 'histories', 'evaluations', 'checkpoints'):
        assert actual[field] is trained[field]
    assert not np.shares_memory(actual['predictions']['scores'], scores)


@pytest.mark.parametrize('field', ['query_keys', 'family_outputs', 'histories', 'evaluations', 'checkpoints'])
def test_incomplete_training_envelope_rejected(field):
    trained = fixture()
    del trained[field]
    with pytest.raises(ValueError, match='training result'):
        adapters.pair_inputs(trained)


def test_mismatched_query_count_rejected():
    trained = fixture()
    context, _, _ = adapters.pair_inputs(trained)
    trained['query_keys'] = ['synthetic-other']
    with pytest.raises(ValueError, match='Query count'):
        adapters.result(trained, context, np.ones(15, dtype=np.float64))


def test_actual_runner_routes_both_generic_sums(monkeypatch, tmp_path):
    import asyncio
    from sciona import decomposed_cassava_pipeline as pipeline
    from sciona.atoms.ml.cassava_execution import cassava_fold_plan
    from sciona.visualizer import runner

    plan = cassava_fold_plan([f'synthetic-{i}' for i in range(25)],
                            [i % 5 for i in range(25)], [i // 5 for i in range(25)])
    controls = dict(source_rows=[], efficientnet_rows=[], query_rows=[], references={},
                    torch_batch_size=1, torch_workers=0, efficientnet_batch_size=1,
                    output_directory=str(tmp_path / 'unused'))
    trained, calls, captured = fixture(), [], {}
    def training(actual_plan, **kwargs):
        assert actual_plan == plan and kwargs == controls
        calls.append(True)
        return trained
    def capture(directory, node, name, value):
        if name.startswith('out_'):
            captured[node, name] = value
    monkeypatch.setattr(pipeline, 'train_families', training)
    monkeypatch.setattr(runner, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(runner, 'save_intermediate_value', capture)
    graph = build_decomposed_cassava_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-cassava', 'case').execute(
        dict(sample_keys=list(plan.keys), labels=list(plan.labels),
             assignments=list(plan.assignments), **controls), cdg=graph))
    assert status['status'] == 'completed'
    assert calls == [True]
    assert set(captured) == {(n.node_id, 'out_' + p.name) for n in graph.nodes for p in n.outputs}
    actual = captured['result', 'out_result']
    expected = assemble(trained['query_keys'], **trained['family_outputs'])
    for field in ('scores', 'labels'):
        np.testing.assert_array_equal(actual['predictions'][field], expected[field])
    assert captured['pair_sum', 'out_paired'].shape == (15,)
    assert captured['final_sum', 'out_scores'].shape == (15,)
