"""Synthetic provider/runner boundaries; training is explicitly replaced here."""
import numpy as np
import pytest

from scripts.build_aptos_execution_graph import build_graph
from scripts.audit_aptos_provider_contract import audit


def test_registered_structure_and_symbolic_contract():
    assert audit()['passed']


@pytest.mark.parametrize('fault', ['optional', 'constraints', 'order', 'provider'])
def test_malformed_graph_rejected(monkeypatch, fault):
    from scripts import audit_aptos_provider_contract as review
    graph = build_graph()
    node = graph.nodes[1]
    if fault == 'optional': node.inputs[-1].required = False
    elif fault == 'constraints': node.inputs[1].constraints = ''
    elif fault == 'order': node.inputs[1], node.inputs[2] = node.inputs[2], node.inputs[1]
    else: node.matched_primitive = 'sciona.atoms.ml.aptos_execution.unregistered'
    monkeypatch.setattr(review, 'build_graph', lambda: graph)
    with pytest.raises(ValueError):
        review.audit()


def test_real_runner_routes_population_and_controls(monkeypatch, tmp_path):
    from scripts.aptos_graph_execution import execute_through_graph
    import sciona.aptos_pipeline as pipeline
    arguments = dict(base=[['synthetic-base'], [4]], average=[['synthetic-average'], [0]],
        grouped=[['synthetic-group'], [2]], pseudo_keys=['synthetic-pseudo'],
        query_keys=['synthetic-pseudo'], images={}, pretrained={}, first_stage_epochs={}, seeds={},
        batch_size=2, learning_rate=1e-4, lower_deviation=.5, upper_deviation=.5,
        tie_policy='upper', work_root=str(tmp_path))
    calls = []
    def run(**kwargs):
        for role in ['base', 'average', 'grouped']:
            assert tuple(kwargs[role][0]) == tuple(arguments[role][0])
            np.testing.assert_array_equal(kwargs[role][1], arguments[role][1])
        assert tuple(kwargs['pseudo_keys']) == tuple(arguments['pseudo_keys'])
        for name in set(arguments) - {'base', 'average', 'grouped', 'pseudo_keys'}:
            assert kwargs[name] == arguments[name]
        calls.append(True)
        return {'synthetic_result': True}
    monkeypatch.setattr(pipeline, 'run_reference', run)
    result = execute_through_graph(**arguments)
    assert result['synthetic_result'] and calls == [True]
    assert result['_graph_evidence']['actual_runner_nodes'] == 2
