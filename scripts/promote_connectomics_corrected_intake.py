"""Promote the complete stored original Connectomics revision at automated Tier 3."""
import argparse
import json
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_connectomics_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='connectomics-corrected-intake-community.v1'
REFERENCE='connectomics-corrected-intake-execution-v1'
DESCRIPTION=('Complete corrected Connectomics simple or tuned precision ensemble with optional temporal precedence blending. '
    'Seven executable nodes bind to four approved providers, including four explicit calls to a reusable ordered circular filter. '
    'One versioned input dictionary supplies signals, mode and directivity; one result dictionary returns scores and fit counts. '
    'All 120 source thresholds, including duplicates, historical PCA variance, filter-specific weights and post-aggregation normalization are preserved. '
    'Synthetic exact-source agreement does not establish biological validity, calibrated probabilities or historical accuracy.')
PLAIN=('Estimate pairwise connection scores from a caller-supplied table of changing signals. '
    'Choose the simple or tuned method and whether to include time order. The wrapped weighted filter can also be used '
    'independently with other arrays, axes and weights. The complete recipe keeps its source-specific choices. '
    'Scores are not a guarantee of real connections or prediction accuracy. This revision preserves the original source record.')
BOUNDS=[
    'Automated Tier 3 only; no Tier 1 human review or Tier 2 community-usage qualification.',
    'Finite nonnegative float32-compatible time-by-node signals, at least three nodes and samples >= max(5,nodes+1); constant channels are rejected.',
    'Caller explicitly chooses simple/tuned and boolean directivity in the version-1 payload. All four circular filters are computed; simple consumes only f1/f2.',
    'Full corrected 120-entry grids retain duplicates and absent 0.153 threshold. Simple performs 240 PCA fits and tuned 480. Weighted precision averaging is not Bayesian inference.',
    'Scalar threshold and filter identity forwarding correct three source bugs. Historical sklearn PCA sample-count variance and 80 percent components are preserved; this is not original buggy-output parity.',
    'Optional strict temporal precedence uses the source endpoint convention and contributes 0.003 after normalization, with precision weight 0.997. Scores are not calibrated probabilities.',
    'Hash-verified provisioned public source and qualified packages are required. Current qualification uses Python 3.13.11, NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0 and joblib 1.5.3.',
    'Nonfinite intermediates, degenerate residual covariance and degenerate score ranges fail closed. Ill-conditioned finite estimates remain a numerical limitation.',
    'Synthetic full-source and stored-runner parity does not establish historical competition accuracy, biological validity or cross-domain validity of the complete recipe.',
    'The numerical circular filter remains domain independent; all source tap choices and scientific interpretation belong to the recipe or caller.',
]


def review():
    from scripts.review_connectomics_filter_providers import review as provider_review
    plan,providers=intake.qualified_plan()
    semantic=provider_review()
    if semantic!=json.loads((ROOT/'docs/reviews/connectomics_filter_provider_semantics.json').read_text()):
        raise ValueError('Provider semantic evidence changed')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        intake.provider_publication.check(db,providers)
    path=ROOT/'docs/reviews/connectomics_corrected_intake_execution.json'
    execution=json.loads(path.read_text())
    if execution['status']!='passed' or execution['stored_candidate'] is not True:
        raise ValueError('Stored source qualification required')
    if (execution['serialized_graph_sha256'],execution['candidate_version_id'],execution['original_artifact_id'])!=(plan['graph_sha256'],plan['version_id'],plan['artifact_id']):
        raise ValueError('Stored execution identity differs')
    expected=[dict(mode=m,directivity=d,pca_fits=240 if m=='simple' else 480,exact_source_match=True)
              for m in ('simple','tuned') for d in (False,True)]
    if execution['cases']!=expected or execution['actual_runner_nodes']!=7:
        raise ValueError('Complete source sweep parity required')
    if not all(execution[k] is True for k in ('all_nodes_uncached','strict_json_output','graph_codec_roundtrip',
        'caller_input_preserved','independent_directivity_exact','degenerate_covariance_rejected','tampered_source_rejected')):
        raise ValueError('Stored execution safety or fidelity gate missing')
    if execution['plan_sha256']!=intake.sha(intake.PLAN):raise ValueError('Stored intake plan changed')
    base=json.loads((ROOT/'docs/reviews/connectomics_decomposed_graph_execution.json').read_text())
    expected_paths=set(base['code_sha256'])-{'scripts/validate_decomposed_connectomics_graph.py'}|{
        'scripts/validate_connectomics_corrected_execution.py','scripts/stage_connectomics_corrected_intake.py',
        'scripts/connectomics_approved_provider_state.py'}
    if set(execution['code_sha256'])!=expected_paths:raise ValueError('Stored implementation evidence incomplete')
    for name,digest in execution['code_sha256'].items():
        if intake.sha(ROOT/name)!=digest:raise ValueError('Stored execution implementation changed')
    for key in ('provider_sha256','circular_provider_sha256'):
        if execution[key]!=base[key]:raise ValueError('Stored provider evidence differs')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/connectomics_filter_provider_semantics.json'),
        limitations=BOUNDS+semantic['limitations'])


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/connectomics_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_connectomics_corrected_publication.py')
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
        url='https://github.com/sciona/sciona/blob/29dcebc4/docs/reviews/connectomics_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed Connectomics execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed Connectomics execution',url=url,
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
        served_cdgs=1,reused_providers=4,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/connectomics_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
