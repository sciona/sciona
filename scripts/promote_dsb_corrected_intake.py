"""Promote the complete stored original DSB revision at automated Tier 3."""
import argparse
import json
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_dsb_corrected_intake as intake
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

ROOT=intake.ROOT
RUNNER='dsb-corrected-intake-community.v1'
REFERENCE='dsb-corrected-intake-execution-v1'
DESCRIPTION=('Explicit source-qualified volumetric preparation, shared detector/classifier training, validation, '
    'model-only checkpoint export and tiled inference. All 19 nodes bind to approved providers; '
    '34 providers include ten reusable numerical operations. Eighteen root inputs expose caller-owned '
    'arrays, checkpoint bytes, geometry, sample-order controls and training options; four outputs retain '
    'the classifier checkpoint, epoch metrics, case score and proposal scores. '
    'Synthetic parity covers two three-epoch windows and final inference, not full competition training or predictive quality.')
PLAIN=('Prepare caller-supplied image volumes, train the source detector and classifier, and produce prediction scores. '
    'The individual numerical steps can be reused in other workflows. The complete recipe requires compatible '
    'model weights, coordinate conventions and training inputs. Scores are not a guarantee of accuracy. '
    'This revision preserves the original source record.')
BOUNDS=[
    'Automated Tier 3 qualification only; no Tier 1 human review or Tier 2 community-usage claim.',
    'Exact qualified CPU environment: Python 3.13.11, NumPy 2.4.2, SciPy 1.17.1, scikit-image 0.26.0 and Torch 2.11.0. Historical environments and clean-install distribution are not qualified.',
    'Pinned-source parity covers classifier variant 3 at epochs 49–51 and variant 4 at epochs 159–161, each with exact checkpoint, RNG, validation and final inference. Full competition training is not claimed.',
    'Explicit detector checkpoint and proposals in the preprocessed coordinate frame are caller inputs. Initial detector training and the alternative res18 architecture are outside this realization.',
    'Caller-provided source arrays, annotations, spatial metadata, sample orders, random streams and callbacks must satisfy the reviewed source contracts. Arbitrary replacement models or factories are not certified.',
    'World-coordinate preparation has full graph evidence; voxel-coordinate preparation has component evidence. Source flips use size-coordinate, training/inference hull limits differ, and classifier rotation leaves coordinates unchanged.',
    'Model-only checkpoints are warm restarts; optimizer and RNG continuity hold inside the consecutive epoch execution, not across restarts.',
    'The ten numerical operations are reusable independently. Model architecture, preprocessing, augmentation, anchor assignment and fixed score thresholds remain recipe-specific.',
    'Synthetic computational agreement does not establish prediction accuracy, calibration, clinical usefulness or cross-domain validity.',
]


def review():
    from scripts import promote_dsb_providers as atoms
    from scripts.qualify_dsb_runtime_environment import environment
    plan,providers=intake.qualified_plan()
    approved=atoms.qualify_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        atoms.check(db,approved,published=True)
    providers['qualification']=approved['qualification']
    path=ROOT/'docs/reviews/dsb_corrected_intake_execution.json';execution=json.loads(path.read_text())
    if not all(execution.get(k) is True for k in ('passed','synthetic_only','stored_candidate')):
        raise ValueError('Stored numerical qualification required')
    if (execution['execution_graph_sha256'],execution['candidate_version_id'])!=(plan['graph_sha256'],plan['version_id']):
        raise ValueError('Stored execution identity differs')
    expected=[dict(variant=variant,start_epoch=start,end_epoch=end,exact_checkpoint_rng_validation=True,
        validation_passes=passes,executed_nodes=19,exact_final_inference=True,proposal_count=count)
        for variant,start,end,passes,count in [(3,49,51,9,77),(4,159,161,7,0)]]
    if execution['cases']!=expected or execution['resource_policy']!=dict(serial=True,native_threads=1,torch_interop_threads=1):
        raise ValueError('Complete source-budget parity required')
    if execution['environment']!=environment():raise ValueError('Qualified environment differs')
    for key,path_to_hash in [('stored_validator_sha256',ROOT/'scripts/validate_dsb_corrected_execution.py'),
                             ('candidate_checker_sha256',ROOT/'scripts/stage_dsb_corrected_intake.py'),
                             ('provider_checker_sha256',ROOT/'scripts/dsb_staged_provider_state.py'),
                             ('retained_environment_evidence_sha256',ROOT/'docs/reviews/dsb_runtime_environment_qualification.json'),
                             ('plan_sha256',intake.PLAN)]:
        if execution[key]!=intake.sha(path_to_hash):raise ValueError('Stored execution evidence drift')
    for path_to_hash,digest in execution['implementation_sha256'].items():
        if intake.sha(ROOT/path_to_hash)!=digest:raise ValueError('Execution implementation changed')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=intake.sha(path),plan_sha256=intake.sha(intake.PLAN),
        provider_review_sha256=intake.sha(ROOT/'docs/reviews/dsb_provider_semantic_review.json'),
        limitations=BOUNDS+providers['qualification']['limitations'])


class InjectedFailure(RuntimeError):pass


def publish(apply=False,fail_at=None):
    plan,providers,qualification=review()
    if apply:
        gate=json.loads((ROOT/'docs/reviews/dsb_corrected_publication_gates.json').read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=intake.sha(__file__)
            or gate['validator_sha256']!=intake.sha(ROOT/'scripts/validate_dsb_corrected_publication.py')
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
        url='https://github.com/sciona/sciona/blob/d7c50534/docs/reviews/dsb_corrected_intake_execution.json'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored decomposed DSB execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored decomposed DSB execution',url=url,
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
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=19,verified_leaf_coverage=1,top_level_input_arity=18,top_level_output_arity=4,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=19,verified_leaf_coverage=1,top_level_input_arity=18,top_level_output_arity=4,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review()[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=34,source_histories_preserved=True,plan_sha256=intake.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['dry','apply','repeat'])
    action=parser.parse_args().action;result=publish(apply=action!='dry')
    if action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    (ROOT/('docs/reviews/dsb_corrected_publication_'+action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
