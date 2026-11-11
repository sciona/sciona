"""Draft expanded tabular lifecycle with explicit fitted-state edges."""
import inspect

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.atoms.ml import tabular_complete_execution as adapters


STAGES = ('prepare', 'fit_stack', 'calibrate', 'predict')


def build_complete_tabular_graph():
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
        publication_status='draft', source_version_ids=['9f3c8c4f-ce85-5e9d-b169-8b055562ce65'],
        scope='Binary mixed-table stacking with explicit feature policies and held-out sigmoid calibration.',
        choices=['Feature cleaning, encodings, aggregates, topics, embeddings and selection fit inside each outer fold.',
            'Logistic and ExtraTrees base models; logistic OOF stacker and full-training base refit.'],
        limitations=['Concrete execution-local states only; forged intermediates are unqualified.',
            'No named competition accuracy, submission format or calibration guarantee.',
            'Caller owns group identity and chronological split policy; large-table resource use unqualified.'],
        num_nodes=len(nodes), num_edges=len(edges)))
