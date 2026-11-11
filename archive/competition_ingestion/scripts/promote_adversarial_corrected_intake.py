"""Promote the complete stored original adversarial revision at automated Tier 3."""
import argparse
import json
import re
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_adversarial_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='adversarial-corrected-intake-community.v1'
REFERENCE='adversarial-corrected-intake-execution-v1'
EVIDENCE_COMMIT='a86c77ec2675558527319c1e0b51769d1a89604c'
DESCRIPTION=('Complete source-derived adversarial image attack with eight-model non-targeted, five-model targeted large-budget and two-model targeted small-budget schedules. '
    'Three executable nodes bind to approved preparation, execution and reusable row-normalization providers. '
    'Normalization is an explicit callable graph dependency invoked at every update. '
    'Source ensemble weights, auxiliary losses, momentum, rounding, clipping, padding and image encoding are preserved in the qualified CPU adaptation. '
    'Synthetic agreement does not establish historical TensorFlow equivalence or pretrained attack quality.')
PLAIN=('Generate changed images using caller-supplied model weights and an explicit attack configuration. '
    'The method repeatedly uses model gradients and keeps changes within the configured bound before image saving. '
    'Its row normalization operation can also be reused independently on numeric tables in other fields. '
    'Saved images use the source scaling rule and do not retain that pixel-change guarantee. '
    'This revision preserves the original source record; tests use synthetic weights and do not measure real attack quality.')
BOUNDS=[
    'Automated Tier 3 only; no Tier 1 human review or Tier 2 usage qualification.',
    'Full 299-by-299 RGB inputs and 1001-class backbones with auxiliary heads; source batch size ten, zero padding and real-entry trimming.',
    'Qualified schedules: eight models and ten non-targeted updates, five models and twenty targeted updates for epsilon >=8, two models and forty targeted updates for epsilon <8.',
    'CPU Torch/NCHW adaptation with sequential model VJPs; no historical TensorFlow binary, reduction-order or max-pool tie-gradient equivalence claim.',
    'Caller supplies explicit initialization states; no implicit downloads, pretrained authenticity or historical accuracy claim. Named tensor mapping is not a TensorFlow checkpoint binary decoder.',
    'Synthetic full states use batch-normalization beta 0.1 to keep padded gradients defined. This is a fixture choice, not a runtime algorithm change.',
    'Normalization divides uncentered rows by mean absolute value or population standard deviation. Zero or nonfinite scales fail without an epsilon stabilizer.',
    'Epsilon projection applies to normalized images only. Source SciPy-style PNG min/max stretching can violate a saved-pixel epsilon bound; constant float images encode black.',
    'The reusable row operation is domain independent; ensemble choices, input layout, recurrence and interpretation remain source recipe or caller policy.',
    'Qualified Python 3.13.11, Torch 2.11.0, NumPy 2.4.2 and Pillow 11.3.0; clean installation is not tested and deployments must configure native thread pools.',
    'Validation uses synthetic inputs, captures intermediates in memory and checks caller image and Torch RNG preservation after provider discovery.',
]


def review():
    from scripts.review_adversarial_normalizer_providers import review as provider_review
    plan,providers=intake.qualified_plan()
    semantic=provider_review()
    if semantic!=json.loads((ROOT/'docs/reviews/adversarial_normalizer_provider_semantics.json').read_text()):
        raise ValueError('Provider semantic evidence changed')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake.provider_publication.check(db,providers)
    path=ROOT/'docs/reviews/adversarial_corrected_intake_execution.json'
    execution=json.loads(path.read_text())
    if execution['result']!='passed' or execution['stored_candidate'] is not True:
        raise ValueError('Stored source qualification required')
    if (execution['serialized_graph_sha256'],execution['candidate_version_id'],execution['original_artifact_id'])!=(plan['graph_sha256'],plan['version_id'],plan['artifact_id']):
        raise ValueError('Stored execution identity differs')
    expected={}
    for branch,models,steps,calls in [('non_targeted',8,10,10),('targeted_large',5,20,40),('targeted_small',2,40,80)]:
        expected[branch]=dict(actual_runner_nodes=3,registered_callable_identity=True,
            registered_normalization_calls=calls,full_models=models,complete_iterations=steps,
            exact_reference_update_checks=steps,neural_batch_size=10,real_entries=1,padded_entries=9,
            only_real_entry_encoded=True,decoded_png_matches_source_scaling=True,
            fixed_projection_bounds=True,input_and_rng_unchanged=True)
    if execution['checks']!=expected:
        raise ValueError('All complete neural schedules and repeated registered callable checks required')
    if execution['plan_sha256']!=intake.sha(intake.PLAN):raise ValueError('Stored intake plan changed')
    base=json.loads((ROOT/'docs/reviews/decomposed_adversarial_graph.json').read_text())
    expected_paths=set(base['sha256'])-{'scripts/validate_decomposed_adversarial_graph.py'}|{
        'scripts/validate_adversarial_corrected_execution.py','scripts/stage_adversarial_corrected_intake.py',
        'scripts/adversarial_approved_provider_state.py'}
    if set(execution['sha256'])!=expected_paths:raise ValueError('Stored implementation evidence incomplete')
    for name,digest in execution['sha256'].items():
        if intake.sha(ROOT/name)!=digest:raise ValueError('Stored execution implementation changed')
    for key in ('execution_provider_sha256','generic_provider_sha256'):
        if execution[key]!=base[key]:raise ValueError('Stored provider evidence differs')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/adversarial_normalizer_provider_semantics.json'),
        limitations=BOUNDS+semantic['limitations'])


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    if not re.fullmatch(r'[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Immutable published execution evidence commit required')
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/adversarial_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_adversarial_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
        if gate['execution_evidence_tests']!=dict(passed=True,count=49,sha256=intake.sha(ROOT/'tests/test_adversarial_corrected_publication.py')):
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
        url=f'https://github.com/sciona/sciona/blob/{EVIDENCE_COMMIT}/docs/reviews/adversarial_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed adversarial execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed adversarial execution',url=url,
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
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=3,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=3,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=3,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/adversarial_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
