"""Actual witness propagation across the complete Andriy graph."""
import importlib
import pytest
from sciona.andriy_execution import build_andriy_execution_graph
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.simulator import SimNode, simulate_graph, PlanError
from sciona.visualizer.runner import get_topo_sorted_leaves


def simulation():
    graph = build_andriy_execution_graph()
    nodes = []
    for node in get_topo_sorted_leaves(graph.nodes, graph.edges, None):
        importlib.import_module(node.matched_primitive.rsplit('.', 1)[0])
        inputs = {port.name: 'inputs/'+port.name for port in node.inputs}
        for edge in graph.edges:
            if edge.target_id == node.node_id:
                inputs[edge.input_name] = edge.source_id+'/'+edge.output_name
        nodes.append(SimNode(name=node.node_id, function_name=node.matched_primitive,
            inputs=inputs, output_names=[node.node_id+'/'+port.name for port in node.outputs]))
    clip = AbstractArray(shape=(16, 153600), dtype='float32')
    initial = {'inputs/'+name: value for name, value in dict(
        csp_candidates=[[clip]*4 for _ in range(3)],
        candidate_sequences=[AbstractArray(shape=(4,), dtype='int64') for _ in range(3)],
        primary_positive=[[clip] for _ in range(3)], auxiliary_positive=[[] for _ in range(3)],
        negative=[[clip] for _ in range(3)], prediction=[[clip]*n for n in [1, 3, 2]],
        sampling_frequencies=[256]*3).items()}
    return nodes, initial


def test_real_witnesses_propagate_population_and_branch_output_shapes():
    nodes, initial = simulation()
    result = simulate_graph(nodes, initial)
    assert result.node_count == 4
    state = result.final_state
    assert [value.shape for value in state['populations/prediction_features']] == [(19, 1965), (57, 1965), (38, 1965)]
    for branch in ['glm', 'svm', 'xgb']:
        assert state[f'{branch}/andriy_{branch}_scores'].shape == (6,)
    assert state['glm/andriy_glm_segment_model_probabilities'].shape == (6, 3)
    assert state['xgb/andriy_xgb_segment_model_probabilities'].shape == (6, 5)
    assert state['svm/andriy_svm_unmasked_window_probabilities'].shape == (114,)


def test_real_witnesses_reject_missing_population():
    nodes, initial = simulation()
    initial['inputs/prediction'] = initial['inputs/prediction'][:2]
    with pytest.raises(PlanError, match='exactly three ordered populations'):
        simulate_graph(nodes, initial)
