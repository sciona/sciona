"""Draft tracking graph with explicit reusable grid interpolation providers."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.physics.regular_grid import fit_linear_grid, evaluate_linear_grid
from sciona.atoms.physics.decomposed_trackml_grid_execution import track
from sciona.decomposed_trackml_residual_graph import build_decomposed_trackml_residual_graph


def build_decomposed_trackml_grid_graph():
    graph = build_decomposed_trackml_residual_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__ + '.' + track.__name__
    tracking.description = inspect.getdoc(track)
    for name, function, output, output_type in [
        ('grid_fit', fit_linear_grid, 'state', 'object'),
        ('grid_evaluate', evaluate_linear_grid, 'result', 'ndarray'),
    ]:
        port = name + '_function'
        tracking.inputs.append(IOSpec(name=port, type_desc='object'))
        inputs = [IOSpec(name=p.name, type_desc='object' if p.name in ('axes', 'state') else 'ndarray')
                  for p in inspect.signature(function).parameters.values()]
        graph.nodes.append(AlgorithmicNode(node_id=name, name=name,
            description=inspect.getdoc(function), concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__+'.'+function.__name__, inputs=inputs,
            outputs=[IOSpec(name=output, type_desc=output_type)]))
        graph.edges.append(DependencyEdge(source_id=name, target_id='track', output_name=output,
            input_name=port, source_type='object', target_type='object', edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_grid_decomposition', num_nodes=24, num_edges=23,
        scope='Configured tracking rounds with twenty-one explicit reusable numeric providers.')
    return graph
