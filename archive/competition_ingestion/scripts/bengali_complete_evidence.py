"""Require immutable complete Bengali execution evidence; grant no approval."""
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'docs/reviews/bengali_complete_graph.json'
PROVIDER = '../sciona-atoms-dl/src/sciona/atoms/dl/grapheme_execution.py'
TRAINING = {
    # The fitter labels epochs with range(epochs); last-of-200 has index 199.
    'train_ood': dict(epochs=200, updates=400, selected_epoch=199, backbone_updated=True),
    'train_seen': dict(epochs=200, updates=400, selected_epoch=199, backbone_updated=True),
    'train_font_a': dict(epochs=60, updates=180),
    'train_font_b': dict(epochs=60, updates=180),
    'train_generator_a': dict(epochs=40, updates=120, all_four_networks_updated=True, frozen_classifier_unchanged=True),
    'train_generator_b': dict(epochs=40, updates=120, all_four_networks_updated=True, frozen_classifier_unchanged=True),
}
REQUIRED_SOFTWARE = {PROVIDER, 'sciona/grapheme_execution.py',
    'sciona/grapheme_workflow_runtime.py', 'sciona/confidence_routing.py',
    'sciona/component_recall.py', 'sciona/svhn_autoaugment.py',
    'scripts/grapheme_synthetic_domains.py', 'scripts/validate_bengali_complete_graph.py',
    'scripts/review_video_numeric_runtime.py', 'sciona/services/execution_graph_codec.py',
    'sciona/visualizer/runner.py'}
RESOURCE_POLICY = dict(compute_threads=1, interop_threads=1, opencv_threads=1, data_loader_workers=0)


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact(actual[k], v) for k, v in expected.items())
    return actual == expected


def validate_report(report):
    expected = dict(format='bengali-complete-graph.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, actual_runner_nodes=22,
        all_nodes_uncached=True, exact_edge_handoffs=32, training=TRAINING,
        inference_checkpoint_replays=8, independent_calibration_scores_match=True,
        final_routing_and_components_match=True, known_training_classes=2,
        font_only_classes=1, calibration_examples=3, query_examples=3,
        resource_policy=RESOURCE_POLICY)
    for key, value in expected.items():
        if not exact(report.get(key), value):
            raise ValueError('Complete execution evidence required: '+key)
    routes = report.get('query_alternate_routes')
    if type(routes) is not int or not 0 <= routes <= 3:
        raise ValueError('Valid query route count required')
    threshold = report.get('fitted_threshold')
    if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Finite bounded calibration threshold required')
    hashes = report.get('implementation_sha256')
    if not isinstance(hashes, dict) or not REQUIRED_SOFTWARE <= hashes.keys():
        raise ValueError('Executed software fingerprints required')
    prerequisites = report.get('prerequisite_sha256')
    if not isinstance(prerequisites, dict) or set(prerequisites) != {
            'docs/reviews/bengali_b7_lifecycle.json', 'docs/reviews/bengali_paired_branches.json'}:
        raise ValueError('Both component prerequisites required')
    for digest in [report.get('graph_sha256'), *hashes.values(), *prerequisites.values()]:
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Exact content fingerprints required')
    limits = report.get('limits')
    if not isinstance(limits, list) or not limits or any(not isinstance(v, str) or not v.strip() for v in limits):
        raise ValueError('Explicit qualification limits required')


def review(commit):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Full immutable execution commit required')
    raw = subprocess.check_output(['git', 'show', commit+':'+REPORT], cwd=ROOT)
    if raw != (ROOT/REPORT).read_bytes():
        raise ValueError('Committed complete execution changed')
    report = json.loads(raw)
    validate_report(report)
    from scripts import validate_bengali_complete_graph as execution
    from sciona.grapheme_execution import build_grapheme_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    required = REQUIRED_SOFTWARE | {str(p.relative_to(ROOT)) for pattern in ('bengali_*.py', 'grapheme_*.py')
                                   for p in (ROOT/'sciona').glob(pattern)}
    if set(report['implementation_sha256']) != required:
        raise ValueError('Complete implementation inventory changed')
    for path, digest in report['implementation_sha256'].items():
        if execution.sha(ROOT/path) != digest:
            raise ValueError('Executed implementation changed: '+path)
    if execution.prerequisites() != report['prerequisite_sha256']:
        raise ValueError('Component prerequisites changed')
    if encode_execution_graph(build_grapheme_graph())[0] != report['graph_sha256']:
        raise ValueError('Executed graph changed')
    checker = execution.runtime_checker
    previous = checker.ROOT_PACKAGES
    try:
        checker.ROOT_PACKAGES = ['numpy', 'torch', 'efficientnet-pytorch', 'pillow', 'scipy',
                                 'opencv-python-headless', 'threadpoolctl']
        if checker.review() != report.get('runtime'):
            raise ValueError('Executed installed runtime changed')
    finally:
        checker.ROOT_PACKAGES = previous
    return dict(passed=True, approved=False, catalog_mutations=0, evidence_commit=commit,
        report_sha256=execution.sha(ROOT/REPORT), graph_sha256=report['graph_sha256'],
        limits=report['limits'], scope='Complete local synthetic execution only; source, reuse, provider and catalog reviews remain separate.')
