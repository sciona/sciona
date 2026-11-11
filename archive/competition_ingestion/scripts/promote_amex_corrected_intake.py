"""Promote the complete stored original amex revision at automated Tier 3."""
import argparse
import json
import re
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_amex_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='amex-corrected-intake-community.v1'
REFERENCE='amex-corrected-intake-execution-v1'
EVIDENCE_COMMIT='775992807851f14fa16ec5af683040b5c8a18494'
DESCRIPTION=('Complete source-derived grouped-sequence classification recipe with five row models, ten downstream tree models and ten neural models. '
    'Ten executable nodes bind to exact approved providers for preparation, population assembly, feature construction, training, prediction assembly and output encoding. '
    'The generic ordered weighted sum receives the source coefficients as explicit data. '
    'All source boosting rounds, neural epochs, feature blocks and prediction branches are preserved. '
    'Synthetic CPU agreement does not establish historical GPU equivalence or competition accuracy.')
PLAIN=('Train the complete recipe on caller-supplied groups and predict scores for query groups. '
    'Keep all feature groups, model branches and training steps. '
    'The numeric row combination can be reused independently with other data and coefficients in other fields. '
    'The recipe keeps its original coefficients without rescaling them. '
    'This revision preserves the original source record; synthetic tests do not measure real prediction quality.')
BOUNDS=[
    'Automated Tier 3 only; no Tier 1 human review or Tier 2 usage qualification.',
    'Five row DART models and ten downstream DART models each evaluate 4500 rounds without active early stopping; empty trees can reduce retained iteration counts.',
    'Ten neural models use ten epochs, hidden width128, batch256 with drop-last and source learning-rate schedule; strict positive metric checkpoint eligibility is required.',
    'All nine manual feature blocks, joint sequence vocabulary, source ranks and thirteen prediction slots are preserved. Joint training/query feature operations are transductive; qualification is not unbiased nested validation.',
    'The four source coefficients .30/.35/.15/.10 are not normalized. Output scores are not claimed to be calibrated probabilities.',
    'The ordered sum is domain independent and supports arbitrary finite coefficients; input interpretation, recipe weights and units remain caller policy.',
    'Intermediate recipe dictionaries must originate from the same invocation. Shape validation and fixed output model-count metadata do not independently authenticate training provenance.',
    'Synthetic complete CPU execution does not establish historical GPU/RNG equivalence, pretrained authenticity or competition accuracy.',
    'Independent mathematical reconstruction; source repository reuse grant remains unresolved. No source repository code, private records or pretrained models are included.',
    'Pinned local software environment; clean installation is untested and deployment must configure native thread pools.',
]


def review():
    from scripts.review_amex_decomposed_providers import review as provider_review
    plan,providers=intake.qualified_plan()
    semantic=provider_review()
    if semantic!=json.loads((ROOT/'docs/reviews/amex_decomposed_provider_semantics.json').read_text()):
        raise ValueError('Provider semantic evidence changed')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake.provider_publication.check(db,providers)
    path=ROOT/'docs/reviews/amex_corrected_intake_execution.json'
    execution=json.loads(path.read_text())
    expected=dict(passed=True,approved=False,catalog_mutations=0,stored_candidate=True,
        actual_runner_nodes=10,edges=16,all_nodes_uncached=True,models=25,dart_models=15,
        rounds_per_dart_model=4500,neural_models=10,epochs_per_neural_model=10,
        exact_reference_output=True,exact_source_blend=True,strict_json=True,input_and_torch_rng_preserved=True)
    if any(execution.get(k)!=v for k,v in expected.items()):
        raise ValueError('Complete stored source computation required')
    if execution.get('format')!='amex-corrected-execution-validation.v1':
        raise ValueError('Stored execution format differs')
    if (execution['serialized_graph_sha256'],execution['candidate_version_id'],execution['original_artifact_id'])!=(plan['graph_sha256'],plan['version_id'],plan['artifact_id']):
        raise ValueError('Stored execution identity differs')
    traces=execution['training_records']
    if set(traces)!={'reference','graph'} or len(traces['reference'])!=15 or traces['reference']!=traces['graph']:
        raise ValueError('Complete matching training traces required')
    for record in traces['graph']:
        if set(record)!={'evaluated_rounds','retained_iterations'} or record['evaluated_rounds']!=4500 or not 1<=record['retained_iterations']<=4500:
            raise ValueError('Complete evaluated boosting rounds required')
    if execution['plan_sha256']!=intake.sha(intake.PLAN):raise ValueError('Stored intake plan changed')
    base=json.loads((ROOT/'docs/reviews/decomposed_amex_graph.json').read_text())
    expected_paths=set(base['implementation_sha256'])-{'scripts/validate_decomposed_amex_graph.py'}|{
        'scripts/validate_amex_corrected_execution.py','scripts/stage_amex_corrected_intake.py',
        'scripts/amex_approved_provider_state.py','docs/reviews/amex_corrected_intake_plan.json',
        'docs/reviews/amex_approved_provider_state.json'}
    if set(execution['implementation_sha256'])!=expected_paths:raise ValueError('Stored implementation evidence incomplete')
    for name,digest in execution['implementation_sha256'].items():
        if intake.sha(ROOT/name)!=digest:raise ValueError('Stored execution implementation changed')
    for key in ('provider_sha256','generic_provider_sha256'):
        if execution[key]!=base[key]:raise ValueError('Stored provider evidence differs')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/amex_decomposed_provider_semantics.json'),
        limitations=BOUNDS+semantic['limitations'])


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    if not re.fullmatch(r'[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Immutable published execution evidence commit required')
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/amex_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_amex_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
        if gate['execution_evidence_tests']!=dict(passed=True,count=33,sha256=intake.sha(ROOT/'tests/test_decomposed_amex_corrected_publication.py')):
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
        url=f'https://github.com/sciona/sciona/blob/{EVIDENCE_COMMIT}/docs/reviews/amex_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed amex execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed amex execution',url=url,
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
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=10,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=10,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=10,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/amex_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
