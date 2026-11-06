"""Publish each fully executed corrected OpenVaccine original at automated Tier 3."""
import argparse
import json
import re
import subprocess
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from scripts import stage_openvaccine_corrected_intakes as staging
from scripts import review_openvaccine_corrected_sources as source_review
from scripts.import_residual_execution_drafts import ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score
from scripts.repair_decomposed_openvaccine_metadata import CONTRACT as PREPARATION_CONTRACT
from scripts.decomposed_openvaccine_provider_contracts import CONTRACTS

ROOT=staging.ROOT
EVIDENCE_COMMIT='ee0a421930366ded164845e6dfe8036b3f54a77e'
TEST_COUNT=95
DESCRIPTION=('Complete source-corrected RNA sequence prediction reconstruction: pinned folding, 55 node and eight adjacency channels, '
    'shared autoencoder pretraining, all twenty TensorFlow GRU/LSTM/forward/wave members, complete caller-configured supervised and '
    'pseudo-label refinement, checkpoint restoration, reverse averaging and corrected all-member clipping. Six exact approved providers '
    'include a domain-independent ordered numeric sum. Compatible dense-feature training supports reuse within fixed dimensions and '
    'architectures. Both original intake identities describe the same solution family and share this implementation. Parallel recurrent/GCN '
    'fusion, PyTorch/PyG, ViennaRNA and original winning clustered-blend claims are excluded. Historical accuracy and training schedule are unproven.')
PLAIN=('Fold RNA sequences and build their numeric features. Train all twenty required models, refine them using generated labels, '
    'and combine their predictions. The numeric sum can combine rows in other fields; the training step accepts compatible graph features '
    'within its stated limits. Two original records share this method. Synthetic tests do not measure real prediction quality.')
PARAMETER_CONTRACT=('The following contract describes the payload validation stage only. After it succeeds, '
    'the complete graph folds populations, trains and refines all members, and combines predictions. '+PREPARATION_CONTRACT)
OUTPUT_CONTRACT=(CONTRACTS['result']['outputs']+
    ' Predictions, teacher_mean and teacher_std are finite float32 values serialized as nested lists '
    'with shape (prediction_population_count,prediction_sequence_length,5). Predictions average clipped '
    'member values in [-0.5,6]; teacher moments use unclipped predictions and the supplied ddof, '
    'with nonnegative teacher_std. members is integer 20, pretraining_steps is the supplied positive '
    'integer, and initial_supervised_steps is twenty times the supplied positive integer. '
    'Each configured round returns integer members=20, supervised_steps=20, pseudo_label_steps=20 '
    'and rollbacks in [0,20]. These counters describe this execution, not historical training fidelity.')


def runner_name(target):
    if target not in staging.TARGETS:raise ValueError('Unknown original identity')
    return 'openvaccine-'+target+'-corrected-intake-community.v1'


def exact(left,right):
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def review(target):
    intake=staging.Intake(target)
    plan,providers=intake.qualified_plan()
    semantic=source_review.review()
    if not exact(semantic,json.loads(source_review.REPORT.read_text())):
        raise ValueError('Source semantic evidence changed')
    with staging.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        staging.provider_publication.check(db,providers)
    path=intake.report_path('execution')
    execution=json.loads(path.read_text())
    expected=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
        stored_candidate=True,all_nodes_uncached=True,actual_imports_verified=True,
        active_dependency_constraints_satisfied=True,target=target,format='openvaccine-corrected-execution.v1')
    if any(not exact(execution.get(k),v) for k,v in expected.items()):
        raise ValueError('Complete stored source computation required')
    expected_checks=dict(actual_runner_nodes=6,actual_runner_edges=7,models=20,refinement_rounds=1,
        pretraining_steps=1,initial_supervised_steps=20,refinement_supervised_steps=20,pseudo_label_steps=20,
        full_reference_retrained=True,exact_reconstruction_result=True,prediction_and_teacher_bytes_equal=True,
        strict_json_output=True,graph_codec_roundtrip=True)
    if not exact(execution['checks'],expected_checks):
        raise ValueError('Complete stored model lifecycle evidence required')
    if not exact(execution['resource_policy'],dict(native_threads=1,tensorflow_intra_threads=1,tensorflow_inter_threads=1,serial_models=True)):
        raise ValueError('Qualified resource policy differs')
    base=json.loads((ROOT/'docs/reviews/decomposed_openvaccine_graph_execution.json').read_text())
    if (execution['serialized_graph_sha256']!=plan['graph_sha256'] or
        execution['candidate_version_id']!=plan['version_id'] or execution['plan_sha256']!=staging.sha(intake.PLAN) or
        execution['bound_provider_sha256']!=base['bound_provider_sha256']):
        raise ValueError('Stored execution identity or provider evidence differs')
    if execution['runtime']!=base['runtime'] or execution['provisioned_software']!=base['provisioned_software']:
        raise ValueError('Stored environment or reviewed software differs')
    expected_paths=set(base['code_sha256'])|{
        'scripts/validate_openvaccine_corrected_execution.py','scripts/stage_openvaccine_corrected_intakes.py',
        'scripts/decomposed_openvaccine_approved_provider_state.py','tests/test_openvaccine_corrected_intakes.py',
        'docs/reviews/openvaccine_approved_provider_state.json',str(intake.PLAN.relative_to(ROOT))}
    if set(execution['code_sha256'])!=expected_paths:
        raise ValueError('Stored implementation evidence incomplete')
    for name,digest in execution['code_sha256'].items():
        if staging.sha(ROOT/name)!=digest:raise ValueError('Stored execution implementation changed')
    return plan,providers,dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        execution_sha256=staging.sha(path),plan_sha256=staging.sha(intake.PLAN),
        source_review_sha256=staging.sha(source_review.REPORT),limitations=semantic['limitations'],
        source_corrections=staging.CORRECTIONS[target],shared_solution_family=True,
        source_reviewer_sha256=staging.sha(source_review.__file__),
        parameter_contract=PARAMETER_CONTRACT,output_contract=OUTPUT_CONTRACT)


class InjectedFailure(RuntimeError):pass


def publish(target,apply=False,fail_at=None):
    intake=staging.Intake(target)
    RUNNER=runner_name(target)
    REFERENCE='openvaccine-'+target+'-corrected-intake-execution-v1'
    if not re.fullmatch(r'[0-9a-f]{40}', EVIDENCE_COMMIT):
        raise ValueError('Immutable published execution evidence commit required')
    evidence_path=str(intake.report_path('execution').relative_to(ROOT))
    for retained in (evidence_path,str(source_review.REPORT.relative_to(ROOT))):
        try:
            committed=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+retained],cwd=ROOT,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            raise ValueError('Immutable execution evidence commit is unavailable') from error
        if committed!=(ROOT/retained).read_bytes():
            raise ValueError('Immutable execution evidence differs from qualified report')
    plan,providers,qualification=review(target)
    if apply:
        gate=json.loads((intake.report_path('publication_gates')).read_text())
        if (not gate['passed'] or not gate['all_rollbacks_complete'] or gate['publisher_sha256']!=staging.sha(__file__)
            or gate['validator_sha256']!=staging.sha(ROOT/'scripts/validate_openvaccine_corrected_publication.py')
            or gate['execution_sha256']!=qualification['execution_sha256']):raise ValueError('Current publication transaction gates required')
        if gate['execution_evidence_tests']!=dict(passed=True,count=TEST_COUNT,sha256=staging.sha(ROOT/'tests/test_openvaccine_corrected_publication.py')):
            raise ValueError('Current execution evidence rejection tests required')
    identity,version=UUID(plan['artifact_id']),UUID(plan['version_id'])
    topology=build_published_cdg_projection(artifact={'artifact_id':str(identity),'fqdn':plan['fqdn']},
        version={'version_id':str(version),'content_hash':plan['graph_sha256']},cdg=intake.build_graph()).topo_hash
    created=updated=0
    with staging.connect() as db:
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
            dejargonized_description='Supply private populations, observed targets, model splits and complete training choices.',
            constraints_json=Jsonb(dict(parameter_contract=PARAMETER_CONTRACT,output_contract=OUTPUT_CONTRACT,
                validity_bounds=qualification['limitations'],review_status='automated_tier_3_pass_with_limits'))))
        url=f'https://github.com/sciona/sciona/blob/{EVIDENCE_COMMIT}/{evidence_path}'
        ensure('references_registry',dict(ref_id=REFERENCE),dict(ref_id=REFERENCE,ref_type='web',title='Full stored corrected OpenVaccine execution',url=url))
        key=dict(artifact_id=identity,ref_key=REFERENCE)
        ensure('artifact_references',key,dict(key,ref_id=REFERENCE,title='Full stored corrected OpenVaccine execution',url=url,
            source='llm_extracted',verified=True,confidence='high',relevance_note='Exact report hash retained in versioned approval evidence; synthetic computation only.'))
        fields=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass_with_limits',
            developer_semantics_status='pass_with_limits',review_status='approved',review_semantic_verdict='pass_with_limits',
            review_developer_semantics_verdict='pass_with_limits',trust_readiness='ready',review_limitations=qualification['limitations'],
            review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits')
        ensure('artifact_audit_rollups',dict(artifact_id=identity),dict(artifact_id=identity,**fields))
        evidence=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=identity,version_id=version,
            audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(qualification=qualification,publisher_sha256=staging.sha(__file__),description=DESCRIPTION,plain_description=PLAIN))))
        if not db.execute('SELECT artifact_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']:
            raise ValueError('Native graph publication requirements missing')
        if not already:
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,leaf_count=6,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,identity)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(identity,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        if fail_at=='activation':raise InjectedFailure('Injected post-activation failure')
        actual=db.execute('SELECT description,leaf_count,verified_leaf_coverage,top_level_input_arity,top_level_output_arity,topo_hash FROM artifacts WHERE artifact_id=%s',(identity,)).fetchone()
        if actual!=dict(description=DESCRIPTION,leaf_count=6,verified_leaf_coverage=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=topology):
            raise ValueError('Published topology or boundaries differ')
        intake.check_candidate(db,plan,providers,approved=True)
        if review(target)[2]!=qualification:raise ValueError('Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        original_artifact_id=str(identity),version_id=str(version),graph_sha256=plan['graph_sha256'],
        served_cdgs=1,reused_providers=6,source_histories_preserved=True,plan_sha256=staging.sha(intake.PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',choices=list(staging.TARGETS))
    parser.add_argument('action',choices=['dry','apply','repeat'])
    args=parser.parse_args();result=publish(args.target,apply=args.action!='dry')
    if args.action=='repeat' and (result['rows_created'] or result['rows_updated']):raise ValueError('Repeat changed rows')
    staging.Intake(args.target).report_path('publication_'+args.action).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
