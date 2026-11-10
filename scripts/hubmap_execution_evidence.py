"""Require committed four-fold HuBMAP execution; no publication authorization."""
import json
import re
import subprocess
from scripts.audit_hubmap_original_readiness import ROOT, sha
from scripts.plasticc_publication_evidence import strict_equal

RESOURCE_POLICY=dict(serial_training=True,torch_threads=1,interop_threads=1,native_threads=1,
    opencv_threads=1,opencv_opencl=False,data_loader_workers=0)
REQUIRED_SOFTWARE={'scripts/validate_hubmap_current_graph.py','sciona/hubmap_lifecycle.py',
    'sciona/hubmap_network.py','sciona/hubmap_encoder.py','sciona/hubmap_graph_runtime.py',
    'sciona/hubmap_retraining.py','sciona/hubmap_raw_inference.py',
    'sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py',
    '../sciona-atoms-dl/src/sciona/atoms/dl/segmentation/hubmap_execution.py'}


def validate_report(report):
    expected=dict(format='hubmap-current-graph.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,all_nodes_uncached=True,actual_runner_nodes=6,full_training_runs=8,
        source_geometry=dict(tile_size=1024,input_side=320,pad_size=256),checkpoint_models_per_stage=4,
        generated_nonempty_pseudo_handoff=True,exact_pseudo_state_routing=True,resource_policy=RESOURCE_POLICY)
    for name,value in expected.items():
        if not strict_equal(report.get(name),value):
            raise ValueError('Complete bounded lifecycle required: '+name)
    folds=report.get('folds')
    wanted=[(stage,fold) for stage in ('initial_training','retraining') for fold in range(4)]
    if not isinstance(folds,list) or len(folds)!=8:
        raise ValueError('Eight fold executions required')
    for row,(stage,fold) in zip(folds,wanted):
        if not isinstance(row,dict):raise ValueError('Fold evidence must be a mapping')
        for key,value in dict(stage=stage,fold=fold,epoch=18,selected_checkpoint='best_loss').items():
            if not strict_equal(row.get(key),value):raise ValueError('Exact ordered fold/epoch/checkpoint required')
        for key in ('optimizer_updates','validation_examples'):
            if type(row.get(key)) is not int or row[key]<1:raise ValueError('Connected training and validation required')
    for key,minimum in [('pseudo_foreground_pixels',1),('final_foreground_pixels',0)]:
        if type(report.get(key)) is not int or not minimum<=report[key]<=600*600:
            raise ValueError('Valid aggregate synthetic mask counts required')
    hashes=report.get('software_sha256',{})
    if not REQUIRED_SOFTWARE<=set(hashes):raise ValueError('Executed software identities required')
    for digest in [*hashes.values(),report.get('prerequisite_sha256'),report.get('graph_sha256')]:
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
            raise ValueError('Exact content fingerprints required')
    limits=report.get('limits')
    if not isinstance(limits,list) or not limits or any(not isinstance(s,str) or not s.strip() for s in limits):
        raise ValueError('Explicit execution limits required')


def review(commit):
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):
        raise ValueError('Full immutable execution commit required')
    path='docs/reviews/hubmap_current_graph.json'
    raw=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
    if raw!=(ROOT/path).read_bytes():raise ValueError('Committed execution evidence changed')
    report=json.loads(raw);validate_report(report)
    for name,digest in report['software_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Executed software changed')
    if sha(ROOT/'docs/reviews/hubmap_current_contracts.json')!=report['prerequisite_sha256']:
        raise ValueError('Component qualification changed')
    from scripts.validate_hubmap_current_graph import prerequisites
    from sciona.hubmap_execution import build_hubmap_lifecycle_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    prerequisites()
    if encode_execution_graph(build_hubmap_lifecycle_graph())[0]!=report['graph_sha256']:
        raise ValueError('Executed graph changed')
    return dict(passed=True,approved=False,catalog_mutations=0,evidence_commit=commit,
        report_sha256=sha(ROOT/path),graph_sha256=report['graph_sha256'],limits=report['limits'],
        scope='Current four-fold local lifecycle only; historical final checkpoint selection, corrected stored execution and publication remain separate.')
