"""Promote the complete stored original Flavours revision at automated Tier 3."""
import argparse
import json
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_flavours_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.review_flavours_provider_semantics import LIMITATIONS
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='flavours-corrected-intake-community.v1'
REFERENCE='flavours-corrected-intake-execution-v1'
DESCRIPTION=('Complete decomposed 115-fit workflow: 35 cross-fit mass regressors, five classifier branches '
    'with 25 fold fits and five full refits, 50 neural fits, six-role nonlinear score blending, and separate '
    'weighted agreement, rank-decorrelation and quality-selected discrimination diagnostics. '
    'All 841 nodes bind to individually approved providers; the closure contains 39 reusable atoms. '
    'Six caller inputs and 51 outputs retain the final report and all 50 neural validation losses. '
    'Caller establishes physical units, source column meanings and independent populations. '
    'Modern CPU reconstruction with complete source budgets; no historical-engine equivalence or predictive-quality guarantee.')
PLAIN=('Train several groups of models, combine their predictions, and report how their scores behave on separate evaluation groups. '
    'The workflow exposes its individual numerical steps for reuse and keeps all training and diagnostic steps visible. '
    'Callers must supply consistently ordered inputs and compatible physical units. '
    'Failed checks remain reported failures. This executable revision preserves the earlier source record.')
BOUNDS=LIMITATIONS+[
    'Full graph recipe expects six physical arrays per population, including 46 base columns and three daughter momenta; generic physical and numerical atoms can be reused separately.',
    'The source excluded column is explicit; corrected feature width is 50 and the ratio offset is 1e-10 in caller mass units.',
    'Neural outer training selections require at least 200 rows of each class. Scaling uses all outer training rows; optimization uses only the inner training split.',
    'Correlation uses 200 neighbours and step 50. Exactly 200 rows produce a vacuous whole-population window; meaningful use needs a larger population.',
    'Query classifier predictions use full refits. Neural contributions are divided before ordered accumulation; no post-sum replacement division.',
    'The catalog output-source map exposes the final result and 50 diagnostic loss scalars. Validation losses never control fitting or model selection.',
]


def review():
    plan,providers=intake.qualified_plan()
    path=ROOT/'docs/reviews/flavours_corrected_intake_execution.json';execution=json.loads(path.read_text())
    required=['passed','synthetic_only','stored_candidate','all_nodes_uncached','reference_and_candidate_both_executed',
              'exact_query_scores','exact_all_population_blends','exact_metrics_and_threshold_flags','exact_population_boundaries','codec_roundtrip']
    if not all(execution.get(k) is True for k in required):raise ValueError('Complete stored numerical qualification required')
    if (execution['graph_sha256'],execution['candidate_version_id'],execution['executed_nodes'],execution['model_fits_per_workflow'],execution['native_threads'])!=(plan['graph_sha256'],plan['version_id'],841,115,1):
        raise ValueError('Stored execution scope differs')
    for key,path_to_hash in [('validator_sha256',ROOT/'scripts/validate_flavours_corrected_execution.py'),
                             ('oracle_sha256',ROOT/'tests/test_flavours_decomposed_evaluation.py'),
                             ('candidate_checker_sha256',ROOT/'scripts/stage_flavours_corrected_intake.py'),
                             ('plan_sha256',intake.PLAN)]:
        if execution[key]!=intake.sha(path_to_hash):raise ValueError('Stored execution evidence drift')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/flavours_provider_semantic_review.json'),limitations=BOUNDS)


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/flavours_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_flavours_corrected_publication.py')
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
        url='https://github.com/sciona/sciona/blob/main/docs/reviews/flavours_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed Flavours execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed Flavours execution',url=url,
            source='llm_extracted',verified=True,confidence='high',relevance_note='Exact report hash retained in versioned approval evidence; synthetic computation only.'))
        fields=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass_with_limits',
            developer_semantics_status='pass_with_limits',review_status='approved',review_semantic_verdict='pass_with_limits',
            review_developer_semantics_verdict='pass_with_limits',trust_readiness='ready',review_limitations=BOUNDS,
            review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits')
        ensure('artifact_audit_rollups',dict(artifact_id=identity),dict(artifact_id=identity,**fields))
        evidence=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=identity,version_id=version,
            audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(qualification=qualification,publisher_sha256=intake.sha(__file__),description=DESCRIPTION,plain_description=PLAIN))))
        if not db.execute('SELECT artifact_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']:
            raise ValueError('Native graph publication requirements missing')
        if not already:
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=841,verified_leaf_coverage=1,top_level_input_arity=6,top_level_output_arity=51,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=841,verified_leaf_coverage=1,top_level_input_arity=6,top_level_output_arity=51,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=39,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/flavours_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
