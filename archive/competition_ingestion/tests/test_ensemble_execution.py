"""Composition checks; upstream numerical validation remains separately scoped."""
import importlib
import inspect
import numpy as np
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer.runner import get_topo_sorted_leaves
from sciona.atoms.riemannian_bci.signal_processing.ensemble_wiring import (
    ensemble_family_inputs, pack_eleven_predictions,
)


def test_full_graph_roundtrip_and_callable_boundaries():
    graph = build_ensemble_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({
        'cdg_nodes': [{**node, 'version_id': 'test'} for node in nodes],
        'cdg_edges': [{**edge, 'version_id': 'test'} for edge in edges]},
        version_id='test', content_hash=digest, require_execution_envelope=True)
    assert restored == graph
    assert len(nodes) == 155 and len(edges) == 523
    assert len(get_topo_sorted_leaves(graph.nodes, graph.edges, None)) == len(nodes)
    consumed = {(edge.target_id, edge.input_name) for edge in graph.edges}
    assert len(consumed) == len(edges)
    roots = []
    for node in graph.nodes:
        module, name = node.matched_primitive.rsplit('.', 1)
        function = getattr(importlib.import_module(module), name)
        parameters = inspect.signature(function).parameters
        supplied = {port.name for port in node.inputs}
        assert supplied <= parameters.keys()
        required = {name for name, p in parameters.items() if p.default is inspect.Parameter.empty}
        assert required <= supplied
        roots.extend(port.name for port in node.inputs if (node.node_id, port.name) not in consumed)
    assert len(roots) == len(set(roots)) == 18


def test_every_model_population_is_collected_with_its_family_identity():
    graph = build_ensemble_execution_graph()
    for model in ['combined', 'autocorrelation', 'coherence', 'relative_power',
                  'feng_xgb', 'feng_knn', 'feng_expanded_knn', 'feng_glm']:
        family = 'feng' if model.startswith('feng_') else 'alex'
        incoming = {e.input_name: e for e in graph.edges if e.target_id == 'collect_'+model}
        for population in range(3):
            assert incoming[f'ids{population}'].source_id == f'{family}_population{population}'
            assert incoming[f'scores{population}'].source_id.startswith(f'{family}_population{population}_')
    pack = next(node for node in graph.nodes if node.node_id == 'pack')
    assert len(pack.inputs) == 14
    assert [p.name for p in pack.inputs[:11]] == ['combined', 'autocorrelation', 'coherence', 'relative_power',
        'feng_xgb', 'feng_knn', 'feng_expanded_knn', 'feng_glm', 'andriy_svm', 'andriy_glm', 'andriy_xgb']


def test_family_boundary_and_model_pack_preserve_distinct_inputs():
    families = [[object(), object(), object()] for _ in range(10)]
    result = ensemble_family_inputs(*families)
    assert all(a is b for a, b in zip(result, families))
    scores = [np.array([float(i)]) for i in range(11)]
    identities = [np.array([i]) for i in range(3)]
    values, ids = pack_eleven_predictions(*scores, *identities)
    assert all(a is b for a, b in zip(values, scores))
    assert [int(value[0]) for value in ids] == [0]*4+[1]*4+[2]*3
