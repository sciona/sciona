"""Execute every selected full Contrails model for one variant with bounded CPU."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import cv2
import torch
from threadpoolctl import threadpool_limits
from sciona import contrails_lifecycle as lifecycle
from sciona.contrails_graph import build_contrails_graph
from sciona.contrails_population import FOLDS
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts.audit_contrails_original_readiness import ROOT, sha
from scripts.validate_contrails_graph import synthetic_payload
from scripts import review_video_numeric_runtime as runtime_checker
from scripts.review_contrails_augmentation_runtime import review as augmentation_review

EVIDENCE = 'docs/reviews/contrails_current_contracts.json'
COMMIT = 'b07e4b40'


def prerequisites():
    raw = (ROOT/EVIDENCE).read_bytes()
    if raw != subprocess.check_output(['git', 'show', COMMIT+':'+EVIDENCE], cwd=ROOT):
        raise ValueError('Immutable component qualification changed')
    report = json.loads(raw)
    if report['passed'] is not True or report['tests'] != dict(tests=67, failures=0, errors=0, skipped=0):
        raise ValueError('Complete component qualification required')
    for path, digest in report['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Qualified software changed')
    previous = runtime_checker.ROOT_PACKAGES
    runtime_checker.ROOT_PACKAGES = report['runtime']['root_packages']
    try:
        if runtime_checker.review() != report['runtime']:
            raise ValueError('Qualified numerical runtime changed')
    finally:
        runtime_checker.ROOT_PACKAGES = previous
    augmentation = json.loads((ROOT/'docs/reviews/contrails_augmentation_runtime.json').read_text())
    if augmentation_review() != augmentation:
        raise ValueError('Qualified augmentation runtime changed')
    return report


def validate(variant):
    baseline = prerequisites()
    paths = list(baseline['software_sha256']) + [
        'scripts/validate_contrails_current_graph.py', 'scripts/validate_contrails_graph.py',
        'scripts/review_contrails_augmentation_runtime.py', 'docs/reviews/contrails_augmentation_runtime.json',
        'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py']
    before = {p:sha(ROOT/p) for p in paths}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if cv2.getNumThreads() != 1 or cv2.ocl.useOpenCL():
        raise ValueError('OpenCV must use one CPU thread without OpenCL')
    digest, nodes, edges = encode_execution_graph(build_contrails_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph codec changed')
    payload = synthetic_payload()
    payload['config']['variant'] = variant
    expected = [(branch, fold) for branch in ('temporal', 'single') for fold in FOLDS[variant][branch]]
    completed = []
    captured = {}
    original = lifecycle.train_fold

    def observed(*args, **kwargs):
        index = len(completed)
        print(json.dumps(dict(variant=variant, fold_number=index+1, status='started')), flush=True)
        trained = original(*args, **kwargs)
        if (len(trained['epochs']) != 1 or trained['initialization'] != 'random'
                or trained['checkpoint_selection'] != 'terminal'
                or trained['epochs'][0]['optimizer_steps'] != 3):
            raise ValueError('Complete configured source fold and terminal checkpoint required')
        completed.append(dict(branch=expected[index][0], fold=expected[index][1],
                              epochs=1, optimizer_updates=3, terminal_checkpoint=True))
        print(json.dumps(dict(variant=variant, fold_number=index+1, status='completed')), flush=True)
        return trained

    def capture(directory, node, name, value):
        if (node, name) in [('prepare', 'out_prepared'), ('execute', 'in_prepared'), ('execute', 'out_result')]:
            captured[node, name] = value

    runner._ensure_atoms_imported()
    with tempfile.TemporaryDirectory(prefix='synthetic-contrails-current-') as temporary, threadpool_limits(limits=1):
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture), \
             patch.object(lifecycle, 'train_fold', side_effect=observed):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-contrails', variant).execute(
                {'payload':payload}, cdg=graph))
    if status['status'] != 'completed':
        raise ValueError('Full graph execution failed: '+str(status.get('error', status['status'])))
    if len(status['trace']) != 2 or {t['node_id'] for t in status['trace']} != {'prepare', 'execute'} or any(t['cached'] for t in status['trace']):
        raise ValueError('Both registered nodes must execute uncached')
    if captured['prepare', 'out_prepared'] is not captured['execute', 'in_prepared']:
        raise ValueError('Prepared state routing changed')
    result = json.loads(json.dumps(captured['execute', 'out_result'], allow_nan=False))
    if (result['variant'] != variant or result['initialization'] != 'random'
            or result['epoch_limit'] != 1 or len(completed) != len(expected)
            or [(f['branch'], f['fold']) for f in result['folds']] != expected
            or any(f['optimizer_updates'] != 3 for f in result['folds'])):
        raise ValueError('Every configured fold must complete')
    values = np.asarray(result['probabilities'])
    if values.shape != (1,1,256,256) or not np.isfinite(values).all() or len(result['masks']) != 1:
        raise ValueError('Complete finite source-sized prediction required')
    if before != {p:sha(ROOT/p) for p in paths} or prerequisites() != baseline:
        raise ValueError('Software changed during execution')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        variant=variant, graph_sha256=digest, software_sha256=before,
        prerequisite_sha256=sha(ROOT/EVIDENCE), folds=completed,
        full_models_completed=len(expected), optimizer_updates=3*len(expected),
        actual_runner_nodes=2, all_nodes_uncached=True, exact_state_routing=True,
        strict_json_output=True, source_sized_inputs_and_outputs=True,
        resource_policy=dict(serial_models=True, torch_threads=1, interop_threads=1,
                             native_threads=1, data_loader_workers=0, opencv_threads=1, opencv_opencl=False),
        limits=['Local serialized graph execution; corrected original stored-graph execution and publication remain separate gates.',
                'One explicitly configured epoch per fold and random initialization; no default 35/40-epoch schedule or pretrained accuracy claim.',
                'All selected full MaxViT/U-Net models, terminal checkpoint reloads and eight-way spatial inference run without stand-ins.',
                'Unnormalized source ensemble sum and fixed threshold retained; no calibration or cross-domain accuracy claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=['v43', 'v47'])
    variant = parser.parse_args().variant
    report = validate(variant)
    (ROOT/('docs/reviews/contrails_current_graph_'+variant+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, variant=variant, models=report['full_models_completed'], approved=False)), flush=True)
