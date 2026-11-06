"""Promote the complete stored original aptos revision at automated Tier 3."""
import argparse
import json
import re
import subprocess
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_aptos_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='aptos-corrected-intake-community.v1'
REFERENCE='aptos-corrected-intake-execution-v1'
EVIDENCE_COMMIT='0d0ca32bad50fc227ee6c1c15017ccb59c09566b'
DESCRIPTION=('Complete source-derived eight-model image regression recipe with two training stages, an ensemble teacher and role-specific target refinement. '
    'Five executable nodes bind to exact approved providers for population preparation, full training, cut policy, generic cut indexing and result assembly. '
    'Four backbone families with two replicas each retain native resolutions, trainable GeM, SmoothL1, checkpoint replay and temporary-file cleanup. '
    'Synthetic CPU qualification does not establish historical winning execution parity or competition accuracy.')
PLAIN=('Train all eight image models in both stages and return scores and class indices for query images. '
    'Keep all input groups, training steps and saved-model checks. '
    'The numeric cut operation can be reused with other values and boundaries in other fields. '
    'The recipe supplies its four fixed cuts and an explicit equality rule. '
    'This revision preserves the original source record; synthetic tests do not measure real prediction quality.')
BOUNDS=[
    'Automated Tier 3 only; no Tier 1 human review or Tier 2 usage qualification.',
    'Four pretrained families, two replicas each. All first-stage fits precede the ensemble teacher; each own checkpoint continues for ten epochs with a fresh Adam optimizer.',
    'First-stage budgets are explicit positive five-epoch increments. The synthetic qualification runs five first-stage epochs and ten second-stage epochs per model.',
    'Base, average, grouped and pseudo roles remain distinct. Query overlap with pseudo inputs is deliberately transductive; supplied-label query overlap is rejected.',
    'Generic ordered cut indexing is domain independent; source cuts [.7,1.5,2.5,3.5] and equality handling are explicit recipe data, with no learned calibration.',
    'Full graph execution checks teacher targets, ensemble scores, class indexing, all fit budgets, checkpoint replay and cleanup. It does not establish historical weight parity or real prediction accuracy.',
    'Inputs and intermediate results may contain private data and must remain private runtime values. Committed evidence is synthetic aggregate qualification only.',
    'Model artifacts are external caller-supplied runtime dependencies; integrity verification does not confer use or redistribution rights.',
    'Pinned local software environment; clean installation is untested and deployment must configure native thread pools.',
]


def review():
    from scripts.review_aptos_decomposed_providers import review as provider_review
    plan,providers=intake.qualified_plan()
    semantic=provider_review()
    if semantic!=json.loads((ROOT/'docs/reviews/aptos_decomposed_provider_semantics.json').read_text()):
        raise ValueError('Provider semantic evidence changed')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake.provider_publication.check(db,providers)
    path=ROOT/'docs/reviews/aptos_corrected_intake_execution.json'
    execution=json.loads(path.read_text())
    expected=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        model_count=8,training_stage_fits=16,first_stage_epochs_per_model=5,
        second_stage_epochs_per_model=10,optimizer_updates=280,native_threads=1,torch_interop_threads=1,
        reference_augmentation_used=True,checkpoint_replays_exact=True,temporary_checkpoints_removed=True,
        target_role_oracle_passed=True,final_ensemble_and_threshold_oracle_passed=True)
    if any(execution.get(k)!=v for k,v in expected.items()):
        raise ValueError('Complete stored source computation required')
    if execution.get('format')!='aptos-corrected-execution-validation.v1':
        raise ValueError('Stored execution format differs')
    base=json.loads((ROOT/'docs/reviews/decomposed_aptos_graph.json').read_text())
    graph_evidence=dict(base['graph_execution'], serialized_graph_sha256=plan['graph_sha256'],
        stored_candidate=True,candidate_version_id=plan['version_id'],plan_sha256=intake.sha(intake.PLAN))
    if execution['graph_execution']!=graph_evidence:
        raise ValueError('Stored execution identity or provider evidence differs')
    if (execution['installed_package_versions']!=base['installed_package_versions'] or
        execution['artifact_sha256']!=base['artifact_sha256']):
        raise ValueError('Stored environment or pretrained artifacts differ')
    expected_paths=set(base['sha256'])-{
        'scripts/validate_decomposed_aptos_graph.py','scripts/decomposed_aptos_graph_execution.py'}|{
        'scripts/validate_decomposed_aptos_corrected_execution.py','scripts/decomposed_aptos_corrected_execution.py',
        'scripts/stage_aptos_corrected_intake.py','scripts/decomposed_aptos_approved_provider_state.py',
        'docs/reviews/aptos_corrected_intake_plan.json','docs/reviews/aptos_approved_provider_state.json'}
    if set(execution['sha256'])!=expected_paths:
        raise ValueError('Stored implementation evidence incomplete')
    for name,digest in execution['sha256'].items():
        if intake.sha(ROOT/name)!=digest:raise ValueError('Stored execution implementation changed')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/aptos_decomposed_provider_semantics.json'),
        limitations=BOUNDS+semantic['limitations'])


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    if not re.fullmatch(r'[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Immutable published execution evidence commit required')
    evidence_path='docs/reviews/aptos_corrected_intake_execution.json'
    try:
        committed=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+evidence_path],cwd=ROOT,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Immutable execution evidence commit is unavailable') from error
    if committed!=(ROOT/evidence_path).read_bytes():
        raise ValueError('Immutable execution evidence differs from qualified report')
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/aptos_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_aptos_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
        if gate['execution_evidence_tests']!=dict(passed=True,count=37,sha256=intake.sha(ROOT/'tests/test_decomposed_aptos_corrected_publication.py')):
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
        url=f'https://github.com/sciona/sciona/blob/{EVIDENCE_COMMIT}/docs/reviews/aptos_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed aptos execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed aptos execution',url=url,
            source='llm_extracted',verified=True,confidence='high',relevance_note='Exact report hash retained in versioned approval evidence; synthetic computation only.'))
        fields=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass_with_limits',
            developer_semantics_status='pass_with_limits',review_status='approved',review_semantic_verdict='pass_with_limits',
            review_developer_semantics_verdict='pass_with_limits',trust_readiness='ready',review_limitations=qualification['limitations'],
            review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits')
        ensure('artifact_audit_rollups',dict(artifact_id=identity),dict(artifact_id=identity,**fields))
        evidence=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=identity,version_id=version,
            audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(qualification=qualification,publisher_sha256=intake.sha(__file__),description=DESCRIPTION,plain_description=PLAIN))))
        if not db.execute('SELECT artifact_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']:
            raise ValueError('Native graph publication requirements missing')
        if not already:
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=5,verified_leaf_coverage=1,top_level_input_arity=15,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=5,verified_leaf_coverage=1,top_level_input_arity=15,top_level_output_arity=1,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=5,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/aptos_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
