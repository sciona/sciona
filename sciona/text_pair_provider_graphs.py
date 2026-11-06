"""Concrete provider graphs for independent numeric/text reuse qualification."""
import inspect
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.atoms.ml.text_nlp.atoms import clean_text, levenshtein
from sciona.atoms.ml.text_nlp.bm25 import score_bm25
from sciona.atoms.ml.sklearn.isotonic.atoms import (
    isotonic_regression, isotonic_regression_fit, isotonic_regression_transform,
)

PROVIDERS = dict(clean=clean_text, edit=levenshtein, bm25=score_bm25,
                isotonic=isotonic_regression, fit=isotonic_regression_fit, transform=isotonic_regression_transform)


def node(name):
    function = PROVIDERS[name]
    signature = inspect.signature(function)
    def annotation(value):
        return value if isinstance(value, str) else inspect.formatannotation(value)
    # Every control, including optional controls, is supplied by the runtime.
    inputs = [IOSpec(name=p.name, type_desc=annotation(p.annotation), required=True,
                     default_value_repr='') for p in signature.parameters.values()]
    return AlgorithmicNode(node_id=name, name=name, description=inspect.getdoc(function) or name,
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive=function.__module__ + '.' + function.__name__, inputs=inputs,
        outputs=[IOSpec(name='result', type_desc=annotation(signature.return_annotation))])


def build_provider_graph(name):
    return CDGExport(nodes=[node(name)], edges=[], metadata=dict(
        scope='Concrete reusable provider qualification; all inputs supplied at runtime.',
        publication_status='draft'))


def build_isotonic_state_graph():
    return CDGExport(nodes=[node('fit'), node('transform')], edges=[DependencyEdge(
        source_id='fit', target_id='transform', output_name='result', input_name='state',
        source_type='IsotonicRegressionState', target_type='IsotonicRegressionState')],
        metadata=dict(scope='Fit isotonic state then transform separately supplied query values.',
                      publication_status='draft'))
