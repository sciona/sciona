"""Sequential lazy graph fallbacks with explicit in-memory terminal values."""
from uuid import uuid4
import numpy as np

from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.services.execution_graph_codec import decode_execution_graph, encode_execution_graph

EMIT = 'sciona.atoms.ml.model_selection.graph_fallbacks.emit'


def graph_descriptor(graph, output_node, output_port):
    digest, nodes, edges = encode_execution_graph(graph)
    result = dict(hash=digest, nodes=nodes, edges=edges, output_node=output_node, output_port=output_port)
    validate_descriptor(result)
    return result


def validate_descriptor(descriptor):
    if not isinstance(descriptor, dict) or set(descriptor) != {'hash','nodes','edges','output_node','output_port'}:
        raise ValueError('Complete hash-bound branch graph required')
    graph = decode_execution_graph(descriptor['nodes'], descriptor['edges'], descriptor['hash'])
    if graph is None:
        raise ValueError('Executable branch graph envelope required')
    selected = [port for node in graph.nodes if node.node_id == descriptor['output_node']
                for port in node.outputs if port.name == descriptor['output_port']]
    if len(selected) != 1 or any(node.node_id == '__emit' for node in graph.nodes):
        raise ValueError('Unique declared branch output required')
    if any(port.name == 'receive' for node in graph.nodes for port in node.inputs):
        raise ValueError('Branch uses reserved receiver input')
    return graph, selected[0]


async def execute_graph_value(descriptor, inputs):
    """Run a verified branch through the existing executor; no global hooks."""
    from sciona.visualizer.runner import CDGExecutionSession
    graph, output = validate_descriptor(descriptor)
    values = []
    graph.nodes.append(AlgorithmicNode(node_id='__emit', name='emit', description='Return branch value to its invoking controller',
        concept_type='data_extraction', status=NodeStatus.ATOMIC, matched_primitive=EMIT,
        inputs=[output.model_copy(update={'name':'value'}), IOSpec(name='receive',type_desc='object',constraints='Controller-owned in-memory receiver.')],
        outputs=[output.model_copy(update={'name':'result'})]))
    graph.edges.append(DependencyEdge(source_id=descriptor['output_node'], target_id='__emit',
        output_name=descriptor['output_port'], input_name='value', source_type=output.type_desc, target_type=output.type_desc))
    result = await CDGExecutionSession(None, 'guarded-prediction-branch', str(uuid4())).execute(
        dict(inputs, receive=values.append), cdg=graph)
    if result['status'] != 'completed' or len(values) != 1:
        raise ValueError('Branch did not return exactly one value')
    return values[0]


async def guarded_prediction(primary, baseline, inputs, count, constant):
    """Try validated graphs lazily; preserve control exceptions and empty rows."""
    if type(count) is not int or count < 0 or not isinstance(inputs, dict):
        raise ValueError('Nonnegative row count and explicit runtime inputs required')
    if type(constant) not in (int,float) or not np.isfinite(constant):
        raise ValueError('Finite scalar constant required')
    # Artifact corruption is not a prediction failure and must not be hidden.
    validate_descriptor(primary)
    validate_descriptor(baseline)
    if count == 0:
        return np.empty(0, dtype=np.float64), 'empty'
    for route, descriptor in [('primary',primary),('baseline',baseline)]:
        try:
            values = await execute_graph_value(descriptor, inputs)
            if not isinstance(values,np.ndarray) or values.shape != (count,) or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
                raise ValueError('Finite row-aligned branch prediction required')
            return values.copy(), route
        except Exception:
            continue
    return np.full(count,float(constant)), 'constant'
