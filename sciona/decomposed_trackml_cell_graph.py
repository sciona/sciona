"""Draft tracking graph with explicit vector selection independent of cell scoring."""
import inspect
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.physics.vector_selection import select_by_absolute_inner_product
from sciona.atoms.physics.decomposed_trackml_cell_execution import track
from sciona.decomposed_trackml_tangent_graph import build_decomposed_trackml_tangent_graph


def build_decomposed_trackml_cell_graph():
    graph = build_decomposed_trackml_tangent_graph()
    tracking = next(n for n in graph.nodes if n.node_id == 'track')
    tracking.matched_primitive = track.__module__+'.'+track.__name__
    tracking.description = inspect.getdoc(track)
    tracking.inputs.append(IOSpec(name='vector_selection_function',type_desc='object'))
    function = select_by_absolute_inner_product
    graph.nodes.append(AlgorithmicNode(node_id='vector_selection',name='vector_selection',
        description=inspect.getdoc(function),concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive=function.__module__+'.'+function.__name__,
        inputs=[IOSpec(name=p.name,type_desc='ndarray') for p in inspect.signature(function).parameters.values()],
        outputs=[IOSpec(name=p,type_desc='ndarray') for p in ('selected','inner')]))
    graph.edges.append(DependencyEdge(source_id='vector_selection',target_id='track',output_name='selected',
        input_name='vector_selection_function',source_type='object',target_type='object',edge_kind=EdgeKind.CALLABLE_INJECTION))
    graph.metadata.update(artifact_source='tracking_cell_decomposition',num_nodes=28,num_edges=27,
        scope='Configured tracking rounds with twenty-five explicit reusable numeric providers.')
    return graph
