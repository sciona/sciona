"""Promote the complete stored original champs revision at automated Tier 3."""
import argparse
import json
import re
import subprocess
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_champs_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='champs-corrected-intake-community.v1'
REFERENCE='champs-corrected-intake-execution-v1'
EVIDENCE_COMMIT='2821ac6f5b54b54a3e9d588d3bd7bad6bc004f30'
DESCRIPTION=('Complete source-derived CHAMPS recipe for raw chemistry, fitted vocabulary and scaling, thirteen full configured graph transformers, checkpoint selection and reload, unscaled per-model prediction and coupling-specific central-five ensemble. '
    'Five executable nodes bind exact approved providers, including a domain-independent central-rank mean. '
    'Synthetic CPU qualification exercises one explicitly requested epoch per model; it does not establish historical training-duration, checkpoint or accuracy parity.')
PLAIN=('Check the input examples and all thirteen model settings, prepare features, train every configured model and reload its saved state before prediction. '
    'Select the source model group for each requested result and average its central five values. '
    'The numeric averaging operation can also be used with other measurements and retained counts in other fields. '
    'This revision preserves the original source record. Synthetic tests do not measure real prediction quality.')
BOUNDS=[
    'Automated Tier 3 only; no Tier 1 human review or Tier 2 usage qualification.',
    'All thirteen full source model architectures and the complete chemistry, training, checkpoint and prediction lifecycle are retained. Training schedules are explicit runtime inputs.',
    'Stored synthetic execution uses one SGD epoch per model with full-population selection. Historical training-duration, pretrained-checkpoint and competition-accuracy parity are unproven.',
    'Generic central-rank mean is domain independent; molecular identities, coupling categories, model masks and retained count five remain explicit recipe policy.',
    'The complete stored graph is executed uncached. All model lifecycles, checkpoint reloads, strict finite JSON output and exact source ensemble arithmetic are checked.',
    'Private runtime chemistry, targets, checkpoints and predictions must remain private. Committed evidence contains synthetic fixtures and aggregate checks only.',
    'Pinned public software is required locally. Clean installation is untested; deployment must configure its own native thread limits.',
]


def review():
    from scripts.review_champs_decomposed_providers import review as provider_review
    plan,providers=intake.qualified_plan()
    semantic=provider_review()
    if semantic!=json.loads((ROOT/'docs/reviews/champs_decomposed_provider_semantics.json').read_text()):
        raise ValueError('Provider semantic evidence changed')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake.provider_publication.check(db,providers)
    path=ROOT/'docs/reviews/champs_corrected_intake_execution.json'
    execution=json.loads(path.read_text())
    expected=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
        stored_candidate=True,all_nodes_uncached=True,epochs_per_model=1,
        format='champs-corrected-execution-validation.v1')
    if any(execution.get(k)!=v for k,v in expected.items()):
        raise ValueError('Complete stored source computation required')
    expected_checks=dict(actual_runner_nodes=5,actual_runner_edges=7,source_ensemble_exact=True,
        full_model_lifecycles=13,synthetic_predictions=12,complete_checkpoint_reload=True,
        strict_json_output=True,graph_codec_roundtrip=True)
    if execution['checks']!=expected_checks:
        raise ValueError('Complete stored model lifecycle evidence required')
    if execution['resource_policy']!=dict(native_threads=1,torch_interop_threads=1,data_loader_workers=0,serial_models=True):
        raise ValueError('Qualified resource policy differs')
    base=json.loads((ROOT/'docs/reviews/decomposed_champs_graph_execution.json').read_text())
    if (execution['serialized_graph_sha256']!=plan['graph_sha256'] or
        execution['candidate_version_id']!=plan['version_id'] or execution['plan_sha256']!=intake.sha(intake.PLAN) or
        execution['provider_sha256']!=base['provider_sha256'] or
        execution['bound_provider_sha256']!=base['bound_provider_sha256']):
        raise ValueError('Stored execution identity or provider evidence differs')
    if execution['runtime']!=base['runtime']:
        raise ValueError('Stored environment differs')
    expected_paths=set(base['code_sha256'])-{'scripts/validate_decomposed_champs_graph_execution.py'}|{
        'scripts/validate_decomposed_champs_corrected_execution.py',
        'scripts/stage_champs_corrected_intake.py','scripts/decomposed_champs_approved_provider_state.py',
        'sciona/services/execution_graph_codec.py',
        'docs/reviews/champs_corrected_intake_plan.json','docs/reviews/champs_approved_provider_state.json'}
    if set(execution['code_sha256'])!=expected_paths:
        raise ValueError('Stored implementation evidence incomplete')
    for name,digest in execution['code_sha256'].items():
        if intake.sha(ROOT/name)!=digest:raise ValueError('Stored execution implementation changed')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/champs_decomposed_provider_semantics.json'),
        limitations=BOUNDS+[s.replace('; original inaccurate intake remains draft.', '; historical inaccurate intake version is retained as unselected provenance.') for s in semantic['limitations']])


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    if not re.fullmatch(r'[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Immutable published execution evidence commit required')
    evidence_path='docs/reviews/champs_corrected_intake_execution.json'
    try:
        committed=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+evidence_path],cwd=ROOT,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Immutable execution evidence commit is unavailable') from error
    if committed!=(ROOT/evidence_path).read_bytes():
        raise ValueError('Immutable execution evidence differs from qualified report')
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/champs_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_champs_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
        if gate['execution_evidence_tests']!=dict(passed=True,count=38,sha256=intake.sha(ROOT/'tests/test_decomposed_champs_corrected_publication.py')):
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
        url=f'https://github.com/sciona/sciona/blob/{EVIDENCE_COMMIT}/docs/reviews/champs_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed champs execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed champs execution',url=url,
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
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=5,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=5,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=topology):
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
    (ROOT/('docs/reviews/champs_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
