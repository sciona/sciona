"""Promote the complete stored original Contrails revision at automated Tier 3."""
from pathlib import Path
import argparse
import json
import re
import subprocess
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_contrails_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='contrails-corrected-intake-community.v1'
REFERENCE='contrails-corrected-intake-execution-v1'
EVIDENCE_COMMIT=None
from scripts import contrails_publication_evidence as evidence
CONTRACTS=json.loads((ROOT/'docs/reviews/contrails_provider_contracts.json').read_text())
PARAMETER_CONTRACT=next(p['constraints'] for p in CONTRACTS['derivative']['ports'] if p['direction']=='input')
OUTPUT_CONTRACT=next(p['constraints'] for p in CONTRACTS['derivative']['ports'] if p['direction']=='output')
SCOPE=json.loads((ROOT/'docs/reviews/contrails_original_scope.json').read_text())
DESCRIPTION='Full two-branch thermal segmentation ensemble with complete selected folds, spatial augmentation, terminal checkpoints and weighted mask decisions. '+' '+SCOPE['reuse_decision']+' '+' '.join(r['realization'] for r in SCOPE['source_stage_review'])
PLAIN='Train image models from ordered thermal frames and caller-supplied targets, then combine spatial predictions into weighted scores and masks. Reuse requires justified physical channel scales, time order, alignment and transformation invariance. Synthetic checks establish execution, not accuracy on a new task.'


def review():
    plan=intake.qualified_plan();providers=None
    qualification=evidence.review()
    qualification.update(plan_sha256=intake.sha(intake.PLAN),parameter_contract=PARAMETER_CONTRACT,output_contract=OUTPUT_CONTRACT,
        evidence_checker_sha256=intake.sha(Path(evidence.__file__)))
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if intake.provider_state(db)!=plan['provider_state']:raise ValueError('Frozen provider state differs')
    return plan,providers,qualification


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    if not isinstance(EVIDENCE_COMMIT,str) or not re.fullmatch(r'[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Immutable published execution evidence commit required')
    if EVIDENCE_COMMIT != evidence.EVIDENCE_COMMIT:
        raise ValueError('Publisher and evidence commit differ')
    for variant in evidence.FOLDS:
        evidence.committed_report('docs/reviews/contrails_corrected_execution_'+variant+'.json')
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/contrails_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(Path(__file__))
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_contrails_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
        if not evidence.strict_equal(gate['qualification'],qualification):
            raise ValueError('Publication qualification changed after rollback checks')
        if gate['execution_evidence_tests']!=dict(passed=True,count=30,sha256=intake.sha(ROOT/'tests/test_contrails_corrected_publication_evidence.py')):
            raise ValueError('Current execution evidence rejection tests required')
    identity,version=UUID(plan['artifact_id']),UUID(plan['version_id'])
    topology=build_published_cdg_projection(artifact={'artifact_id':str(identity),'fqdn':plan['fqdn']},
        version={'version_id':str(version),'content_hash':plan['graph_sha256']},cdg=intake.build_catalog_contrails_graph()).topo_hash
    created=updated=0
    with intake.connect() as db:
        for lock in (intake.RUNNER,RUNNER):db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(lock,))
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',(identity,)).fetchone()
        already=state==dict(status='approved',is_publishable=True)
        intake.check(db,plan,approved=already)
        def ensure(table,keys,row):
            nonlocal created
            created+=int(ensure_row(db,table,keys,row))
            if fail_at=='metadata' and created>=3:raise InjectedFailure('Injected metadata failure')
        for kind,content in [('technical',DESCRIPTION),('dejargonized',PLAIN)]:
            key=dict(artifact_id=identity,kind=kind,language='en')
            ensure('artifact_descriptions',key,dict(key,content=content,generated_by=RUNNER,reviewed=False,
                jargon_score=round(estimate_jargon_score(content),6)))
        for position,port in enumerate(p for p in plan['ports'] if p['direction']=='input'):
            key=dict(artifact_id=identity,version_id=version,name=port['name'])
            ensure('artifact_parameters',key,dict(key,type_desc=port['type_desc'],required=port['required'],
                default_value_repr=port['default_value_repr'],position=position,kind='positional_or_keyword',
                technical_description=port['constraints'],dejargonized_description='Supply '+port['name']+' according to the explicit thermal encoding, temporal alignment, target meanings, fold populations and training controls.',
                constraints_json=Jsonb(dict(parameter_contract=port['constraints'],output_contract=OUTPUT_CONTRACT,
                    validity_bounds=qualification['limitations'],review_status='automated_tier_3_pass_with_limits'))))
        for variant in evidence.FOLDS:
            reference=REFERENCE+'-'+variant
            url=f'https://github.com/sciona/sciona/blob/{EVIDENCE_COMMIT}/docs/reviews/contrails_corrected_execution_{variant}.json'
            title='Full stored thermal segmentation execution '+variant
            ensure('references_registry',dict(ref_id=reference),dict(ref_id=reference,ref_type='web',title=title,url=url))
            key=dict(artifact_id=identity,ref_key=reference)
            ensure('artifact_references',key,dict(key,ref_id=reference,title=title,url=url,
                source='llm_extracted',verified=True,confidence='high',relevance_note='Exact report hash retained in versioned approval evidence; synthetic computation only.'))
        fields=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass_with_limits',
            developer_semantics_status='pass_with_limits',review_status='approved',review_semantic_verdict='pass_with_limits',
            review_developer_semantics_verdict='pass_with_limits',trust_readiness='ready',review_limitations=qualification['limitations'],
            review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits')
        ensure('artifact_audit_rollups',dict(artifact_id=identity),dict(artifact_id=identity,**fields))
        evidence_id=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence_id),dict(evidence_id=evidence_id,artifact_id=identity,version_id=version,
            audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(qualification=qualification,publisher_sha256=intake.sha(Path(__file__)),description=DESCRIPTION,plain_description=PLAIN))))
        if not db.execute('SELECT artifact_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']:
            raise ValueError('Native graph publication requirements missing')
        if not already:
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=2,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=2,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check(db,plan,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=2,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/contrails_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
