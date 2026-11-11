"""Direct registered numeric graphs without physical units or recipe choices."""
import inspect
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.atoms.ml.constrained_arrays import project_nonnegative_fixed_sum, smooth_periodic_rows

PROVIDERS = dict(project=project_nonnegative_fixed_sum, smooth=smooth_periodic_rows)


def node(name):
    fn = PROVIDERS[name]
    signature = inspect.signature(fn)
    return AlgorithmicNode(node_id=name, name=name, description=inspect.getdoc(fn),
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive=fn.__module__+'.'+fn.__name__,
        inputs=[IOSpec(name=p.name, type_desc=inspect.formatannotation(p.annotation), required=True)
            for p in signature.parameters.values()],
        outputs=[IOSpec(name='result', type_desc=inspect.formatannotation(signature.return_annotation))])


def build_provider_graph(name):
    return CDGExport(nodes=[node(name)], edges=[], metadata=dict(publication_status='draft',
        scope='Reusable numeric operation; finite arrays and explicit parameters.'))


def build_composed_graph():
    return CDGExport(nodes=[node('project'), node('smooth')], edges=[DependencyEdge(
        source_id='project', target_id='smooth', output_name='result', input_name='values',
        source_type='numpy.ndarray', target_type='numpy.ndarray')], metadata=dict(
        publication_status='draft', scope='Explicit fixed-sum projection followed by circular smoothing.'))
