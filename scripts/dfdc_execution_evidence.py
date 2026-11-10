"""Require immutable complete DFDC graph execution before provider qualification."""
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
REQUIRED_SOFTWARE={'scripts/validate_dfdc_current_graph.py','scripts/validate_dfdc_hull_backend.py',
    'scripts/validate_dfdc_training.py','sciona/dfdc_runtime.py','sciona/dfdc_classifier.py',
    'sciona/dfdc_selection.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py',
    '../sciona-atoms-dl/src/sciona/atoms/dl/dfdc_execution.py'}


def validate_report(report):
    expected=dict(format='dfdc-current-graph.v1',result='passed',approved=False,catalog_mutations=0,
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


def review(commit):
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Full immutable execution commit required')
    path='docs/reviews/dfdc_current_graph.json'
    raw=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
    if raw!=(ROOT/path).read_bytes():
        raise ValueError('Committed execution evidence changed')
    report=json.loads(raw)
    validate_report(report)
    for name,digest in report['software_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Executed software changed')
    for name,digest in report['prerequisite_sha256'].items():
        if sha(ROOT/('docs/reviews/'+name+'.json'))!=digest:
            raise ValueError('Prerequisite report changed')
    from scripts.validate_dfdc_current_graph import prerequisites
    from sciona.dfdc_graph import build_dfdc_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    prerequisites()
    if encode_execution_graph(build_dfdc_graph())[0]!=report['serialized_graph_sha256']:
        raise ValueError('Executed graph changed')
    return dict(passed=True,approved=False,catalog_mutations=0,evidence_commit=commit,
        report_sha256=sha(ROOT/path),graph_sha256=report['serialized_graph_sha256'],
        limits=report['limits'],scope='Full local graph only; corrected stored original and publication remain separate.')
