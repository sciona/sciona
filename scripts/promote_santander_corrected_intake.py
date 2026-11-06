"""Promote the complete stored original santander revision at automated Tier 3."""
import argparse
import json
import re
import subprocess
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_santander_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='santander-corrected-intake-community.v1'
REFERENCE='santander-corrected-intake-execution-v1'
EVIDENCE_COMMIT='b7c436cf86777fcd1f3d8c1e8a513f0e2ded0e15'
DESCRIPTION=('Complete independent supervised-uniqueness neural/tree pseudo-label ensemble: initial neural folds, both full hard-label tails, branch-specific feature regeneration, neural/tree refits, average-tie ranks and explicit 2.1:1 blend. '
    'Seven executable nodes use exact approved providers, including a domain-independent ordered weighted sum. '
    'Global feature preparation and query-dependent scaling prevent independent validation claims. Historical winning implementation and accuracy are not reproduced.')
PLAIN=('Check private training and query rows. Train initial models, select both full high and low score tails, train both branches again and combine their ranks. '
    'The separate weighted sum can combine numeric rows in other fields. '
    'Keep all query outputs and selection counts. These are rank scores, not probabilities; synthetic tests do not measure real prediction quality.')
BOUNDS=[
    'Automated Tier 3 community qualification only; no Tier 1 human certification, Tier 2 usage qualification or performance guarantee.',
    'Correct the incomplete tree-only intake to the complete initial-neural, pseudo-label, neural/tree retraining lifecycle. Preserve original identity, immutable intake history and existing derivative.',
    'Query-only singleton detection informs uniqueness statistics while all query outputs remain. Other-row supervised class presence and mean substitution replace the inaccurate raw count interpretation.',
    'Global supervised feature preparation precedes folds; neural scaling uses training plus all query rows. Pseudo-selected query rows remain in the unlabeled reference. Validation scores are not independent generalization estimates.',
    'Retain ten folds per stage, fifteen neural epochs, every caller seed, full neural tails 5000/3000 and tree tails 2700/2000. Reject ambiguous score ties and insufficient query capacity. Tree controls, categorical handling, pseudo-reference membership and framework randomness remain explicit independent choices.',
    'Stored synthetic execution fits thirty models over four feature groups and 8500 query rows with full tail counts, every node uncached, and complete output equality to the existing reconstruction. Historical feature width, training volume and competition strength are not established.',
    'The generic ordered sum is reusable across domains. Average-tie ranking, neural/tree weights 2.1 and 1, and post-sum division by 3.1 remain explicit recipe policy. Scores are not calibrated probabilities.',
    'Same-invocation intermediate provenance is required but not authenticated by structural checks. Private inputs, learner states and outputs remain private.',
    'Provisioned package evidence does not establish clean deployment or historical source-code equivalence. Deployment must configure its own resource limits.',
]


def review():
    from scripts.review_santander_decomposed_providers import review as provider_review
    plan,providers=intake.qualified_plan()
    semantic=provider_review()
    if semantic!=json.loads((ROOT/'docs/reviews/santander_decomposed_provider_semantics.json').read_text()):
        raise ValueError('Provider semantic evidence changed')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake.provider_publication.check(db,providers)
    path=ROOT/'docs/reviews/santander_corrected_intake_execution.json'
    execution=json.loads(path.read_text())
    expected=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
        stored_candidate=True,all_nodes_uncached=True,
        format='santander-corrected-execution-validation.v1')
    if any(execution.get(k)!=v for k,v in expected.items()):
        raise ValueError('Complete stored source computation required')
    expected_checks=dict(actual_runner_nodes=7,actual_runner_edges=13,models_fitted=30,
        synthetic_query_rows=8500,full_default_pseudo_counts=True,exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True)
    if execution['checks']!=expected_checks:
        raise ValueError('Complete stored model lifecycle evidence required')
    if execution['resource_policy']!=dict(native_threads=1,lightgbm_threads=1,torch_threads=1,torch_interop_threads=1,serial_models=True):
        raise ValueError('Qualified resource policy differs')
    base=json.loads((ROOT/'docs/reviews/decomposed_santander_graph_execution.json').read_text())
    if (execution['serialized_graph_sha256']!=plan['graph_sha256'] or
        execution['candidate_version_id']!=plan['version_id'] or execution['plan_sha256']!=intake.sha(intake.PLAN) or
        execution['bound_provider_sha256']!=base['bound_provider_sha256']):
        raise ValueError('Stored execution identity or provider evidence differs')
    if execution['runtime']!=base['runtime']:
        raise ValueError('Stored environment differs')
    expected_paths=set(base['code_sha256'])-{'scripts/validate_decomposed_santander_graph_execution.py'}|{
        'scripts/validate_decomposed_santander_corrected_execution.py',
        'scripts/stage_santander_corrected_intake.py','scripts/decomposed_santander_approved_provider_state.py',
        'sciona/services/execution_graph_codec.py',
        'docs/reviews/santander_corrected_intake_plan.json','docs/reviews/santander_approved_provider_state.json'}
    if set(execution['code_sha256'])!=expected_paths:
        raise ValueError('Stored implementation evidence incomplete')
    for name,digest in execution['code_sha256'].items():
        if intake.sha(ROOT/name)!=digest:raise ValueError('Stored execution implementation changed')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/santander_decomposed_provider_semantics.json'),
        limitations=BOUNDS+semantic['limitations'])


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    if not re.fullmatch(r'[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Immutable published execution evidence commit required')
    evidence_path='docs/reviews/santander_corrected_intake_execution.json'
    try:
        committed=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+evidence_path],cwd=ROOT,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Immutable execution evidence commit is unavailable') from error
    if committed!=(ROOT/evidence_path).read_bytes():
        raise ValueError('Immutable execution evidence differs from qualified report')
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/santander_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_santander_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
        if gate['execution_evidence_tests']!=dict(passed=True,count=39,sha256=intake.sha(ROOT/'tests/test_decomposed_santander_corrected_publication.py')):
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
        url=f'https://github.com/sciona/sciona/blob/{EVIDENCE_COMMIT}/docs/reviews/santander_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed santander execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed santander execution',url=url,
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
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=7,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=7,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=7,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/santander_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
