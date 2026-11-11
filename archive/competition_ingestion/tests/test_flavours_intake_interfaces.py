"""Intake ports must match real call signatures and all graph occurrences."""
import pytest
from sciona.flavours_decomposed_evaluation import build_evaluated_graph
from sciona.atoms.ml.model_selection.configured_crossfit import shuffled_kfold_masks
from scripts.plan_flavours_provider_intake import interfaces


def node():
    return next(n for n in build_evaluated_graph().nodes if n.matched_primitive.endswith('.shuffled_kfold_masks'))


def test_intake_preserves_call_parameter_order_and_output_ordinals():
    n=node();n.inputs=list(reversed(n.inputs))
    ports=interfaces(shuffled_kfold_masks,[n])
    assert [p['name'] for p in ports if p['direction']=='input']==['features','n_splits','seed','fold_index']
    assert [(p['name'],p['ordinal']) for p in ports if p['direction']=='output']==[('fit',0),('held',1)]


@pytest.mark.parametrize('fault',['missing_input','wrong_type','output_arity','repeated_output_names'])
def test_intake_rejects_signature_or_repeated_graph_contract_drift(fault):
    original=node();changed=original.model_copy(deep=True)
    if fault=='missing_input':changed.inputs=changed.inputs[:-1]
    elif fault=='wrong_type':changed.inputs[0].type_desc='dict'
    elif fault=='output_arity':changed.outputs=changed.outputs[:-1]
    else:changed.outputs[0].name='wrong'
    with pytest.raises(ValueError):interfaces(shuffled_kfold_masks,[original,changed])
