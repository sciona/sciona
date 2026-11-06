"""Promote the complete stored original trackml revision at automated Tier 3."""
import argparse
import json
import re
import subprocess
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_trackml_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='trackml-corrected-intake-community.v1'
REFERENCE='trackml-corrected-intake-execution-v1'
EVIDENCE_COMMIT='701d27cef32ac9d402333c6d41dd4056b74a31b8'
DESCRIPTION=intake.SCOPE+' '+ ' '.join(intake.CORRECTIONS.values())
PLAIN='Build tracks from supplied observations and detector geometry. Fit curves, search nearby observations, apply supplied correction maps, rank candidate tracks and assign observations. Numerical operations can also be selected separately for other fields. The detector layout and tracking rules remain specific to this recipe; synthetic checks do not measure real tracking quality.'
from scripts import trackml_publication_evidence as evidence
PARAMETER_CONTRACT='Versioned runtime payload decoded by sciona.trackml_payload.decode_tracking_payload. Caller supplies detector geometry, complete event observations, settings and optional cells and layer maps. Finite numeric table values and supported integer representations are required. The source detector topology, enough observations for every configured layer query, consistent coordinate units and complete configured tracking rounds are mandatory. Optional maps are applied, not trained. Cell processing requires positive weights and valid module rotations. Runtime input values are never bundled with the graph. Exact numeric function contracts are carried by the 25 independently bound providers.'
OUTPUT_CONTRACT='In-memory encoded assignments aligned to the complete input event, including unassigned observations. No file persistence, truth scoring, historical accuracy, global optimum or calibrated physical validity claim. See sciona.trackml_payload.encode_tracking_result and the retained source comparison limits.'


def review():
    plan,providers=intake.qualified_plan()
    qualification=evidence.review()
    qualification.update(plan_sha256=intake.sha(intake.PLAN),parameter_contract=PARAMETER_CONTRACT,output_contract=OUTPUT_CONTRACT,
        evidence_checker_sha256=intake.sha(evidence.__file__))
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake.provider_publication.check(db,providers)
    return plan,providers,qualification


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    if not re.fullmatch(r'[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Immutable published execution evidence commit required')
    evidence_path='docs/reviews/trackml_corrected_intake_execution.json'
    try:
        committed=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+evidence_path],cwd=ROOT,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Immutable execution evidence commit is unavailable') from error
    if committed!=(ROOT/evidence_path).read_bytes():
        raise ValueError('Immutable execution evidence differs from qualified report')
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/trackml_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_trackml_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
        if not evidence.strict_equal(gate['qualification'],qualification):
            raise ValueError('Publication qualification changed after rollback checks')
        if gate['execution_evidence_tests']!=dict(passed=True,count=10,sha256=intake.sha(ROOT/'tests/test_trackml_publication_evidence.py')):
            raise ValueError('Current execution evidence rejection tests required')
    identity,version=UUID(plan['artifact_id']),UUID(plan['version_id'])
    topology=build_published_cdg_projection(artifact={'artifact_id':str(identity),'fqdn':plan['fqdn']},
        version={'version_id':str(version),'content_hash':plan['graph_sha256']},cdg=intake.build_graph()).topo_hash
    created=updated=0
    with intake.connect() as db:
        for lock in (intake.RUNNER,RUNNER):db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(lock,))
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',(identity,)).fetchone()
        already=state==dict(status='approved',is_publishable=True)
        intake.check_candidate(db,plan,providers,approved=already)
        def ensure(table,keys,row):
            nonlocal created
            created+=int(ensure_row(db,table,keys,row))
            if fail_at=='metadata' and created>=3:raise InjectedFailure('Injected metadata failure')
        for kind,content in [('technical',DESCRIPTION),('dejargonized',PLAIN)]:
            key=dict(artifact_id=identity,kind=kind,language='en')
            ensure('artifact_descriptions',key,dict(key,content=content,generated_by=RUNNER,reviewed=False,
                jargon_score=round(estimate_jargon_score(content),6)))
        key=dict(artifact_id=identity,version_id=version,name='payload')
        ensure('artifact_parameters',key,dict(key,type_desc='dict',required=True,
            default_value_repr='',position=0,kind='positional_or_keyword',
            technical_description=PARAMETER_CONTRACT,
            dejargonized_description='Supply observations, detector geometry, explicit tracking settings and optional correction maps and cells. Detector assumptions remain part of this recipe.',
            constraints_json=Jsonb(dict(parameter_contract=PARAMETER_CONTRACT,output_contract=OUTPUT_CONTRACT,
                validity_bounds=qualification['limitations'],review_status='automated_tier_3_pass_with_limits'))))
        url=f'https://github.com/sciona/sciona/blob/{EVIDENCE_COMMIT}/docs/reviews/trackml_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed trackml execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed trackml execution',url=url,
            source='llm_extracted',verified=True,confidence='high',relevance_note='Exact report hash retained in versioned approval evidence; synthetic computation only.'))
        fields=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass_with_limits',
            developer_semantics_status='pass_with_limits',review_status='approved',review_semantic_verdict='pass_with_limits',
            review_developer_semantics_verdict='pass_with_limits',trust_readiness='ready',review_limitations=qualification['limitations'],
            review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits')
        ensure('artifact_audit_rollups',dict(artifact_id=identity),dict(artifact_id=identity,**fields))
        evidence_id=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence_id),dict(evidence_id=evidence_id,artifact_id=identity,version_id=version,
            audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(qualification=qualification,publisher_sha256=intake.sha(__file__),description=DESCRIPTION,plain_description=PLAIN))))
        if not db.execute('SELECT artifact_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']:
            raise ValueError('Native graph publication requirements missing')
        if not already:
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=28,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=28,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=28,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/trackml_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
