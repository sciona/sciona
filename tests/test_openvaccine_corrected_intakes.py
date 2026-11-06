"""Reject cross-identity plans and incomplete immutable provenance before writes."""
import copy
import json
import pytest
from scripts import stage_openvaccine_corrected_intakes as stage


@pytest.fixture(scope='module')
def providers():
    return stage.provider_publication.qualify_plan()


@pytest.fixture
def candidate(monkeypatch,tmp_path,providers):
    monkeypatch.setattr(stage.provider_publication,'qualify_plan',lambda:providers)
    intake=stage.Intake('covid')
    plan=json.loads(intake.PLAN.read_text())
    intake.PLAN=tmp_path/'plan.json'
    return intake,plan


def write_plan(intake,plan):
    intake.PLAN.write_text(json.dumps(plan))


def test_distinct_identities_share_full_computation(monkeypatch,providers):
    monkeypatch.setattr(stage.provider_publication,'qualify_plan',lambda:providers)
    left,right=[stage.Intake(k) for k in stage.TARGETS]
    a,b=left.build_graph(),right.build_graph()
    assert a.nodes==b.nodes and a.edges==b.edges
    assert len(a.nodes)==6 and len(a.edges)==7
    assert a.metadata['original_source_version_id']!=b.metadata['original_source_version_id']
    assert a.metadata['source_corrections']==stage.CORRECTIONS['covid']
    assert b.metadata['source_corrections']==stage.CORRECTIONS['degradation']
    assert stage.encode_execution_graph(a)[0]!=stage.encode_execution_graph(b)[0]
    assert left.boundaries(a)==right.boundaries(b)


def test_valid_plan_preserves_alias_and_both_sources(candidate):
    intake,plan=candidate
    write_plan(intake,plan)
    actual,_=intake.qualified_plan()
    assert len(actual['dependencies'])==9
    blend=next(b for b in actual['bindings'] if b['node_id']=='blend')
    assert blend['evidence_summary']['output_aliases_by_ordinal']==['total']
    assert sum(d['dependency_role']=='logic_atom' for d in actual['dependencies'])==6


@pytest.mark.parametrize('fault',[
    'artifact_id','fqdn','original_version_id','original_content_hash','version_id',
    'binding_hash','binding_alias','binding_runtime','missing_other_source',
    'optional_source','duplicate_dependency','wrong_provider_version',
    'wrong_graph','wrong_builder','wrong_base_evidence','wrong_provider_plan',
])
def test_corrupt_plan_rejected_before_database(candidate,fault):
    intake,original=candidate
    plan=copy.deepcopy(original)
    if fault in ('artifact_id','fqdn','original_version_id','original_content_hash','version_id'):
        plan[fault]='wrong-identity'
    elif fault=='binding_hash':plan['bindings'][0]['bound_version_content_hash']='0'*64
    elif fault=='binding_alias':plan['bindings'][-2]['evidence_summary']['output_aliases_by_ordinal']=['scores']
    elif fault=='binding_runtime':plan['bindings'][0]['evidence_summary']['runtime_fqdn']='wrong.provider'
    elif fault=='missing_other_source':
        plan['dependencies']=[d for d in plan['dependencies'] if d['dependency_artifact_fqdn']!=intake.other['fqdn']]
    elif fault=='optional_source':plan['dependencies'][0]['optional']=True
    elif fault=='duplicate_dependency':plan['dependencies'].append(copy.deepcopy(plan['dependencies'][0]))
    elif fault=='wrong_provider_version':plan['dependencies'][-1]['binding_metadata']['provider_version_id']='wrong-version'
    else:
        field={'wrong_graph':'graph_sha256','wrong_builder':'builder_sha256',
            'wrong_base_evidence':'qualified_base_execution_sha256',
            'wrong_provider_plan':'provider_publication_plan_sha256'}[fault]
        plan[field]='0'*64
    write_plan(intake,plan)
    with pytest.raises(ValueError):intake.qualified_plan()
