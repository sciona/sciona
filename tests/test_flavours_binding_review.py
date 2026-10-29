"""Binding review rejects contract corruption while retaining native defaults."""
from copy import deepcopy
import json

import pytest

from sciona.atoms.ml.constrained_ml.decorrelation.atoms import roc_auc_truncated_weighted
from sciona.flavours_decomposed_evaluation import build_evaluated_graph
from scripts.reconcile_flavours_legacy_drafts import ports
from scripts.review_flavours_decomposed_bindings import check_call, ROOT
from scripts.complete_flavours_provider_contracts import expected_dependencies


def auc_case():
    node=next(n for n in build_evaluated_graph().nodes if n.node_id=='discrimination_metric')
    return node,ports(roc_auc_truncated_weighted)


def test_omitted_auc_defaults_are_explicitly_retained():
    node,catalog=auc_case()
    assert check_call(node,roc_auc_truncated_weighted,catalog)==dict(
        omitted_native_defaults={'tpr_thresholds':'(0.2, 0.4, 0.6, 0.8)','weights':'(4, 3, 2, 1, 0)'},output_aliases=[])


@pytest.mark.parametrize('fault',['missing_required','input_type','output_alias','output_type','output_ordinal','default','required_flag'])
def test_binding_review_rejects_corrupt_contracts(fault):
    node,catalog=auc_case()
    if fault=='missing_required':node.inputs=node.inputs[1:]
    elif fault=='input_type':node.inputs[0].type_desc='dict'
    elif fault=='output_alias':node.outputs[0].name='accuracy'
    elif fault=='output_type':node.outputs[0].type_desc='int'
    elif fault=='output_ordinal':catalog[-1]['ordinal']=1
    elif fault=='default':next(p for p in catalog if p['name']=='weights')['default_value_repr']='(1, 1, 1, 1, 1)'
    else:catalog[0]['required']=False
    with pytest.raises(ValueError):check_call(node,roc_auc_truncated_weighted,catalog)


def test_dependencies_pin_required_exact_target_versions_and_hashes():
    records=json.loads((ROOT/'docs/reviews/flavours_provider_post_reconciliation_inventory.json').read_text())['providers']
    dependencies=expected_dependencies(records)
    assert len(dependencies)==11
    assert all(not d['optional'] and d['dependency_role']=='logic_atom' for d in dependencies)
    caller=next(r for r in records if r['runtime_fqdn'].endswith('.standard_scaler_fit'))
    target=next(r for r in records if r['runtime_fqdn'].endswith('.standard_scaler_partial_fit'))
    dependency=next(d for d in dependencies if d['dependent_version_id']==caller['version_id'])
    assert dependency['dependency_content_hash']==target['content_hash']
    assert dependency['binding_metadata']['provider_version_id']==target['version_id']
    broken=deepcopy(records)
    broken.remove(next(r for r in broken if r['runtime_fqdn']==target['runtime_fqdn']))
    with pytest.raises(KeyError):expected_dependencies(broken)
