"""Require immutable complete DFDC graph execution before original publication."""
import json
import re
import subprocess
from scripts.audit_dfdc_original_readiness import ROOT, sha
from scripts.plasticc_publication_evidence import strict_equal

RESOURCE_POLICY=dict(serial_models=True,torch_threads=1,interop_threads=1,native_threads=1,
    opencv_threads=1,opencv_opencl=False,data_loader_workers=0)
CHECKS=dict(json_input_roundtrip=True,actual_MTCNN_preparation_and_prediction=True,
    actual_native_synthetic_predictor_loaded=True,full_B7_training_runs=5,
    connected_training_updates=10,validation_epochs=10,selected_ensemble_models=7,
    nonfallback_prediction=True,caller_rng_restored=True,runtime_path_and_states_excluded_from_output=True)
REQUIRED_SOFTWARE={'scripts/validate_dfdc_corrected_execution.py','scripts/stage_dfdc_corrected_intake.py',
    'sciona/catalog_dfdc_graph.py','scripts/validate_dfdc_hull_backend.py',
    'scripts/validate_dfdc_training.py','sciona/dfdc_runtime.py','sciona/dfdc_classifier.py',
    'sciona/dfdc_selection.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py',
    '../sciona-atoms-dl/src/sciona/atoms/dl/dfdc_execution.py'}


def validate_report(report):
    expected=dict(format='dfdc-corrected-execution.v1',stored_graph_verified_before_and_after=True,result='passed',approved=False,catalog_mutations=0,
        synthetic_only=True,all_nodes_uncached=True,exact_state_routing=True,
        checks=CHECKS,resource_policy=RESOURCE_POLICY)
    for name,value in expected.items():
        if not strict_equal(report.get(name),value):
            raise ValueError('Complete bounded graph execution required: '+name)
    for name,minimum in [('training_records',1),('classified_faces',2)]:
        value=report.get('aggregate',{}).get(name)
        if type(value) is not int or value<minimum:
            raise ValueError('Nonfallback synthetic population required')
    if not REQUIRED_SOFTWARE<=set(report.get('software_sha256',{})):
        raise ValueError('Executed software identities required')
    if set(report.get('prerequisite_sha256',{}))!={'dfdc_current_contracts','dfdc_full_runtime'}:
        raise ValueError('Component and full runtime prerequisites required')
    if not isinstance(report.get('limits'),str) or not report['limits'].strip():
        raise ValueError('Explicit execution limits required')


# Filled only after completed stored execution is committed.
EVIDENCE_COMMIT = 'f6d86579c99ca7ff9294136b4ea3daa0892baeaa'
EXECUTION = 'docs/reviews/dfdc_corrected_execution.json'
SOURCE = 'docs/reviews/dfdc_original_scope.json'


def committed_report(path):
    if not isinstance(EVIDENCE_COMMIT,str) or not re.fullmatch('[0-9a-f]{40}',EVIDENCE_COMMIT):
        raise ValueError('Full immutable stored execution commit required')
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+path],cwd=ROOT)
    if raw!=(ROOT/path).read_bytes():
        raise ValueError('Committed qualification changed')
    return json.loads(raw)


def validate_stored_report(report,plan):
    validate_report(report)
    for key,expected in [('stored_version_id',plan['version_id']),
                         ('serialized_graph_sha256',plan['graph_sha256']),
                         ('staging_plan_sha256',plan['plan_sha256'])]:
        if not strict_equal(report.get(key),expected):
            raise ValueError('Stored execution identity differs: '+key)


def qualification_limits(source,providers,report):
    groups=[source['limitations'],providers['limitations'],[report['limits']]]
    if any(not isinstance(group,list) or not group or
           any(not isinstance(item,str) or not item.strip() for item in group)
           for group in groups):
        raise ValueError('Explicit limits required from every qualification layer')
    return list(dict.fromkeys(item for group in groups for item in group))


def review():
    from scripts import stage_dfdc_corrected_intake as intake
    from scripts.validate_dfdc_corrected_execution import prerequisites
    report=committed_report(EXECUTION)
    source=committed_report(SOURCE)
    providers=committed_report('docs/reviews/dfdc_provider_contracts.json')
    plan=intake.qualified_plan()
    validate_stored_report(report,dict(plan,plan_sha256=sha(intake.PLAN)))
    if source['artifact_id']!=intake.IDENTITY or source['source_version_id']!=intake.ORIGINAL:
        raise ValueError('Reviewed original differs')
    for path,digest in report['software_sha256'].items():
        if sha(ROOT/path)!=digest:
            raise ValueError('Executed software changed')
    for name,digest in report['prerequisite_sha256'].items():
        if sha(ROOT/('docs/reviews/'+name+'.json'))!=digest:
            raise ValueError('Prerequisite qualification changed')
    prerequisites()
    return dict(review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        evidence_commit=EVIDENCE_COMMIT,execution_sha256=sha(ROOT/EXECUTION),
        source_review_sha256=sha(ROOT/SOURCE),limitations=qualification_limits(source,providers,report),
        resolved_source_corrections={r['stage_id']:r['realization']+' '+r['correction']
                                    for r in source['source_stage_review']},
        reuse_decision=source['reuse_decision'])
