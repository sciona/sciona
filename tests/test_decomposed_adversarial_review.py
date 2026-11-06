"""Reject incomplete or stale full-graph evidence before any catalog mutation."""
import json
from pathlib import Path

import pytest

from scripts import review_decomposed_adversarial as review


@pytest.mark.parametrize('field,value', [
    ('actual_runner_nodes', 2), ('full_models', 1), ('complete_iterations', 1),
    ('exact_reference_update_checks', 0), ('registered_normalization_calls', 0),
    ('neural_batch_size', 1), ('real_entries', 10), ('padded_entries', 0),
    ('registered_callable_identity', False), ('only_real_entry_encoded', False),
    ('decoded_png_matches_source_scaling', False), ('fixed_projection_bounds', False),
    ('input_and_rng_unchanged', False),
])
def test_rejects_missing_graph_guarantee(monkeypatch, field, value):
    target = review.ROOT/'docs/reviews/decomposed_adversarial_graph.json'
    original = Path.read_text
    document = json.loads(original(target))
    document['checks']['targeted_small'][field] = value
    def corrupted(path, *args, **kwargs):
        return json.dumps(document) if path == target else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', corrupted)
    with pytest.raises(ValueError, match='coverage differs|invariant failed'):
        review.review()


def test_rejects_stale_graph_identity(monkeypatch):
    original = review.encode_execution_graph
    def changed(graph):
        _, nodes, edges = original(graph)
        return 'unqualified', nodes, edges
    monkeypatch.setattr(review, 'encode_execution_graph', changed)
    with pytest.raises(ValueError, match='Qualified graph changed'):
        review.review()


def test_callable_parameters_are_not_root_inputs():
    graph = review.build_decomposed_adversarial_graph()
    interfaces = {n.node_id: {'input': n.inputs, 'output': n.outputs} for n in graph.nodes}
    boundary = review.boundary_ports(graph, interfaces)
    assert [p.name for p in boundary['input']] == ['payload']
    assert [p.name for p in boundary['output']] == ['result']
