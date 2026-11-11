"""Draft full separate-field lifecycle with explicit execution-state edges."""
import inspect

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.atoms.ml import text_pair_complete_execution as adapters


STAGES = ('prepare_fields', 'normalize_fields', 'fit_features', 'fit_models', 'predict', 'format_rows')


def build_complete_text_pair_graph():
    nodes, edges = [], []
    for name in STAGES:
        fn = getattr(adapters, name)
        signature = inspect.signature(fn)
        parameter = next(iter(signature.parameters.values()))
        nodes.append(AlgorithmicNode(node_id=name, name=name.replace('_', ' '),
            description=inspect.getdoc(fn), concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=fn.__module__ + '.' + fn.__name__,
            inputs=[IOSpec(name=parameter.name, type_desc=inspect.formatannotation(parameter.annotation))],
            outputs=[IOSpec(name='result', type_desc=inspect.formatannotation(signature.return_annotation))]))
        if len(nodes) > 1:
            edges.append(DependencyEdge(source_id=nodes[-2].node_id, target_id=name,
                output_name='result', input_name=parameter.name, source_type='object', target_type='object'))
    return CDGExport(nodes=nodes, edges=edges, metadata=dict(
        publication_status='draft', source_version_ids=['1169253d-76d3-5b76-885f-07b58bed795c'],
        scope='Separate title/description binary relevance with explicit normalization and numerical controls.',
        limitations=['Concrete execution-local states only; forged intermediates are unqualified.',
            'Calibration selection F1 is not independent accuracy.',
            'Generic ordered rows, not a named competition submission format.'],
        num_nodes=len(nodes), num_edges=len(edges)))
