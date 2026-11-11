"""Draft decomposed physical-operator lifecycle with explicit fitted-state edges."""
import inspect

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.atoms.ml import physical_operator_complete_execution as adapters


STAGES = ('prepare', 'fit', 'predict', 'project', 'finish')


def build_complete_physical_operator_graph():
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
        publication_status='draft', source_version_ids=['49c645b7-3659-5cfa-a386-78f7a51f7358'],
        scope='Periodic scalar diffusion with explicit SI units, trained Fourier/local blocks and fixed-sum postprocessing.',
        choices=['Uniform endpoint-excluded grids; caller-supplied separated group identities.',
            'CPU float64 full-batch Adam; earliest minimum validation-MSE checkpoint.',
            'Nonnegative initial-sum projection followed by explicit periodic smoothing.'],
        limitations=['Concrete execution-local states only; forged intermediates are unqualified.',
            'No exact PDE integration, arbitrary PDE, resolution transfer or empirical accuracy guarantee.',
            'Validation selects checkpoints; its score is not independent final accuracy.',
            'Provider catalog qualification and corrected-original publication remain pending.'],
        num_nodes=len(nodes), num_edges=len(edges)))
