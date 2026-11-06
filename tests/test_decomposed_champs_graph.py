"""Full-body preservation and real runner routing; training is mocked here."""
import ast
import asyncio
import importlib
import inspect
from pathlib import Path

from sciona.atoms.ml import decomposed_champs_execution as adapters
from sciona.champs_ensemble import MODEL_ORDER, blend_predictions
from sciona.champs_graph import build_champs_graph
from sciona.decomposed_champs_graph import build_decomposed_champs_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph


def test_complete_training_body_preserved():
    root = Path(__file__).resolve().parents[1] / 'sciona'
    old = ast.parse((root / 'champs_runtime.py').read_text())
    new = ast.parse((root / 'decomposed_champs_runtime.py').read_text())
    original = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    actual = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'train_models')
    original.name = actual.name
    returned = original.body[-1].value
    position = next(i for i, key in enumerate(returned.keys) if key.value == 'predictions')
    assert ast.unparse(returned.values[position]) == 'blend_predictions(predictions, prepared.inference_types)'
    returned.keys[position:position + 1] = [ast.Constant('model_predictions'), ast.Constant('coupling_types')]
    returned.values[position:position + 1] = [ast.Name(id='predictions', ctx=ast.Load()),
        ast.Attribute(value=ast.Name(id='prepared', ctx=ast.Load()), attr='inference_types', ctx=ast.Load())]
    assert ast.dump(original, include_attributes=False) == ast.dump(actual, include_attributes=False)


def test_graph_roots_signatures_and_codec():
    graph, original = build_decomposed_champs_graph(), build_champs_graph()
    assert len(graph.nodes) == 5 and len(graph.edges) == 7
    assert graph.nodes[0] == original.nodes[0]
    assert graph.nodes[1].inputs == original.nodes[1].inputs
    connected = {(e.target_id, e.input_name) for e in graph.edges}
    assert [(n.node_id, p.name) for n in graph.nodes for p in n.inputs
            if (n.node_id, p.name) not in connected] == [('prepare', 'payload')]
    for node in graph.nodes:
        module, name = node.matched_primitive.rsplit('.', 1)
        function = getattr(importlib.import_module(module), name)
        assert list(inspect.signature(function).parameters) == [p.name for p in node.inputs]
    for key in ('source_version_id', 'source_commits', 'scope', 'exclusions', 'source_corrections'):
        assert graph.metadata[key] == original.metadata[key]
    digest, nodes, edges = encode_execution_graph(graph)
    assert decode_execution_graph(nodes, edges, digest) == graph


def test_runner_routes_complete_envelope_through_generic_mean(monkeypatch, tmp_path):
    from sciona import decomposed_champs_runtime as pipeline
    from sciona.visualizer import runner
    from tests.test_champs_contract import payload
    predictions = {name: {'synthetic-request': float(i)} for i, name in enumerate(MODEL_ORDER)}
    kinds = {'synthetic-request': '2JHH'}
    # Empty histories are intentional: this test does not qualify training.
    trained = dict(version=1, execution_version='synthetic-boundary', model_predictions=predictions,
                   coupling_types=kinds, models_completed=list(MODEL_ORDER), training={})
    calls, captured = [], {}
    def training(prepared):
        assert prepared.inference_types == kinds
        assert set(prepared.model_options) == set(MODEL_ORDER)
        calls.append(True)
        return trained
    def capture(directory, node, name, value):
        if name.startswith('out_'):
            captured[node, name] = value
    monkeypatch.setattr(pipeline, 'train_models', training)
    monkeypatch.setattr(runner, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(runner, 'save_intermediate_value', capture)
    graph = build_decomposed_champs_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-champs', 'case').execute(
        {'payload': payload()}, cdg=graph))
    assert status['status'] == 'completed'
    assert calls == [True]
    assert set(captured) == {(n.node_id, 'out_' + p.name) for n in graph.nodes for p in n.outputs}
    actual = captured['result', 'out_result']
    assert actual == dict(version=1, execution_version='synthetic-boundary',
                          predictions=blend_predictions(predictions, kinds),
                          models_completed=list(MODEL_ORDER), training={})
    assert actual['training'] is trained['training']


def test_result_rejects_wrong_invocation_keys():
    import numpy as np
    import pytest
    trained = dict(version=1, execution_version='synthetic', model_predictions={},
                   coupling_types={'synthetic-a': '2JHH'}, models_completed=[], training={})
    with pytest.raises(ValueError, match='identities'):
        adapters.result(trained, ('synthetic-b',), np.array([1.]))
