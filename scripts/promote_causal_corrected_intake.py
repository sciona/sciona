"""Promote the complete stored original cause-effect revision at automated Tier 3."""
import argparse
import json
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_causal_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='causal-corrected-intake-community.v1'
REFERENCE='causal-corrected-intake-execution-v1'
DESCRIPTION=('Explicit typed-pair feature extraction, five independently fitted gradient-boosting classifiers, '
    'label-aware probability selection and three-system score fusion. '
    'All 114 nodes bind to approved providers; the transitive closure contains 32 atoms. '
    'The 43-feature ordering and classifier role labels belong to this recipe. Generic statistics, '
    'weighted fitting and probability prediction remain separately reusable across domains. '
    'Eleven root inputs expose paired samples, type declarations, labels, weights and model controls. '
    'Synthetic pinned-source parity includes five 500-estimator fits; no predictive-quality or causal-validity guarantee.')
PLAIN=('Compute measurements from pairs of variables, train five models, and combine their scores. '
    'The individual numerical steps can be reused in other workflows. Callers must supply compatible '
    'variable types, training labels and consistently ordered pairs. A score alone does not establish cause and effect. '
    'This revision preserves the original source record.')
BOUNDS=[
    'Automated Tier 3 qualification only; no Tier 1 human review or Tier 2 community-usage claim.',
    'Synthetic same-library parity with pinned upstream code; no historical-library reproduction or predictive-quality claim.',
    'The recipe fixes 43 selected statistics, adjacent forward/reverse observations and sign-reversed labels.',
    'Training labels require all roles: one-step {-1,0,1}, direction {-1,1}, binary dependence/left/right {0,1}.',
    'Each classifier fits independently. Query observations do not enter fitting. Meaningful training populations remain a caller responsibility.',
    'Source defaults use 500 estimators, depth 9, minimum split 8, learning rate 0.1 and seed 1. Direction uses P(+1), not a signed score.',
    'Undefined statistic regimes are rejected. Scalar conventions and supported parameter modes retain the individual provider review bounds.',
    'Generic numerical atoms may be reused separately; feature ordering, type dispatch and model-role adapters retain recipe-specific contracts.',
]


def review():
    plan,providers=intake.qualified_plan()
    path=ROOT/'docs/reviews/causal_corrected_intake_execution.json';execution=json.loads(path.read_text())
    if not all(execution.get(k) is True for k in ('passed','synthetic_only','stored_candidate')):
        raise ValueError('Stored numerical qualification required')
    if (execution['graph_hash'],execution['candidate_version_id'],execution['nodes'],execution['edges'])!=(plan['graph_sha256'],plan['version_id'],114,571):
        raise ValueError('Stored execution scope differs')
    expected=[dict(seed=seed,estimators_per_fit=n,uncached_nodes=114,maximum_absolute_error=0.0) for seed,n in [(731,12),(732,12),(733,500)]]
    if execution['cases']!=expected or execution['resource_policy']!=dict(serial=True,native_threads=1):
        raise ValueError('Complete uncached source-budget parity required')
    for key,path_to_hash in [('validator_sha256',ROOT/'scripts/validate_causal_corrected_execution.py'),
                             ('candidate_checker_sha256',ROOT/'scripts/stage_causal_corrected_intake.py'),
                             ('retained_base_evidence_sha256',ROOT/'docs/reviews/causal_full_decomposition_source_parity.json'),
                             ('plan_sha256',intake.PLAN)]:
        if execution[key]!=intake.sha(path_to_hash):raise ValueError('Stored execution evidence drift')
    for path_to_hash,digest in execution['implementation_hashes'].items():
        if intake.sha(ROOT/path_to_hash)!=digest:raise ValueError('Execution implementation changed')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/causal_provider_semantic_review.json'),
        limitations=BOUNDS+providers['qualification']['limitations'])


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/causal_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_causal_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
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
        url='https://github.com/sciona/sciona/blob/38fa059c/docs/reviews/causal_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed cause-effect execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed cause-effect execution',url=url,
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
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=114,verified_leaf_coverage=1,top_level_input_arity=11,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=114,verified_leaf_coverage=1,top_level_input_arity=11,top_level_output_arity=1,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=32,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/causal_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
