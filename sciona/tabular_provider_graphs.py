"""Direct numeric provider graphs, independent of tabular recipe policy."""
import inspect
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.atoms.ml.tabular.gradient_boosting.atoms import (
    frequency_encode_fit, frequency_encode, target_encode, group_aggregate,
)
from sciona.atoms.ml.sklearn.decomposition.lda_shell.atoms import lda_normalize_document_topics
from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping.atoms import rfe_active_feature_indices

PROVIDERS = dict(frequency_fit=frequency_encode_fit, frequency_apply=frequency_encode,
    target=target_encode, aggregate=group_aggregate, normalize=lda_normalize_document_topics,
    indices=rfe_active_feature_indices)


def node(name):
    fn = PROVIDERS[name]; signature = inspect.signature(fn)
    def annotation(value): return value if isinstance(value, str) else inspect.formatannotation(value)
    return AlgorithmicNode(node_id=name, name=name, description=inspect.getdoc(fn),
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=fn.__module__+'.'+fn.__name__,
        inputs=[IOSpec(name=p.name, type_desc=annotation(p.annotation), required=True, default_value_repr='')
            for p in signature.parameters.values()],
        outputs=[IOSpec(name='result', type_desc=annotation(signature.return_annotation))])


def build_provider_graph(name):
    return CDGExport(nodes=[node(name)], edges=[], metadata=dict(publication_status='draft',
        scope='Reusable numeric provider; all optional arguments explicitly supplied.'))
