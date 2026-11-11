"""Run the complete ensemble's actual registered shape witnesses."""
import importlib
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.simulator import SimNode, simulate_graph
from sciona.visualizer.runner import get_topo_sorted_leaves


def test_full_ensemble_actual_witness_propagation():
    graph = build_ensemble_execution_graph()
    nodes = []
    for node in get_topo_sorted_leaves(graph.nodes, graph.edges, None):
        importlib.import_module(node.matched_primitive.rsplit('.', 1)[0])
        inputs = {port.name: 'inputs/'+port.name for port in node.inputs}
        for edge in graph.edges:
            if edge.target_id == node.node_id:
                inputs[edge.input_name] = edge.source_id+'/'+edge.output_name
        nodes.append(SimNode(name=node.node_id, function_name=node.matched_primitive,
            inputs=inputs, output_names=[node.node_id+'/'+port.name for port in node.outputs]))
    raw = AbstractArray(shape=(16, 240000), dtype='float32')
    transposed = AbstractArray(shape=(240000, 16), dtype='float32')
    labels = AbstractArray(shape=(4,), dtype='int64')
    ids = AbstractArray(shape=(2,), dtype='int64')
    values = {}
    for family, clip in [('alex', raw), ('feng', transposed)]:
        values.update({family+'_training_populations': [[clip]*4 for _ in range(3)],
            family+'_prediction_populations': [[clip]*2 for _ in range(3)],
            family+'_label_populations': [labels]*3, family+'_identity_populations': [ids]*3})
    values.update(csp_candidates=[[raw]*4 for _ in range(3)],
        candidate_sequences=[AbstractArray(shape=(4,), dtype='int64')]*3,
        primary_positive=[[raw]*13 for _ in range(3)], auxiliary_positive=[[raw] for _ in range(3)],
        negative=[[raw]*14 for _ in range(3)], prediction=[[raw]*2 for _ in range(3)],
        sampling_frequencies=[400]*3, andriy_identity_populations=[ids]*3,
        output_ids=AbstractArray(shape=(6,), dtype='int64'), baseline=AbstractArray(shape=(6,), dtype='float64'))
    result = simulate_graph(nodes, {'inputs/'+name: value for name, value in values.items()})
    assert result.node_count == 155
    assert result.final_state['blend/ensemble_scores'].shape == (6,)
    assert len(result.final_state['pack/predictions']) == 11
    assert all(value.shape == (6,) for value in result.final_state['pack/predictions'])
