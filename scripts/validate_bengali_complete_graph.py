"""Execute the full decoded training/calibration/query graph on zero-shot fixtures."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from scripts.grapheme_synthetic_domains import complete_populations
from scripts import review_video_numeric_runtime as runtime_checker
from sciona.bengali_population import ImagePopulation
from sciona.grapheme_execution import build_grapheme_graph
from sciona.grapheme_routing_calibration import submission_components
from sciona.grapheme_workflow_runtime import predict_b7, predict_generated
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT.parent/'sciona-atoms-dl/src/sciona/atoms/dl/grapheme_execution.py'
PREREQUISITES = {
    'docs/reviews/bengali_b7_lifecycle.json': '6ecd92c7701d10ac2296fe629165f102a9922f44',
    'docs/reviews/bengali_paired_branches.json': '3cd27c42ae4a6ddeb92ec985c2dc2ef5e6c7e925',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prerequisites():
    result = {}
    for name, commit in PREREQUISITES.items():
        raw = (ROOT/name).read_bytes()
        if raw != subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT):
            raise ValueError('Committed component evidence differs')
        report = json.loads(raw)
        if report['passed'] is not True or report['approved'] is not False or report['catalog_mutations'] != 0:
            raise ValueError('Passing component evidence required')
        for path, digest in report['implementation_sha256'].items():
            if sha(ROOT/path) != digest:
                raise ValueError('Qualified component implementation changed: '+path)
        result[name] = sha(ROOT/name)
    return result


def inputs_for(data, runtime, checkpoint):
    hand, calibration = data['hand'], data['calibration']
    known_ids = np.flatnonzero(data['known_class_mask'])
    selected = data['known_class_mask'][calibration.labels]
    known_validation = ImagePopulation(calibration.images[selected], calibration.labels[selected])
    inputs = dict(validation_images=calibration.images, query_images=data['query'].images,
        truth_components=submission_components(calibration.labels), known_class_mask=data['known_class_mask'],
        sample_weight=np.ones(len(calibration.images)), thresholds=np.linspace(0., 1., 21))
    for index, branch in enumerate(('ood', 'seen')):
        options = dict(epochs=200, batch_size=2, seed=3200+index*100, mean=[.5]*3, std=[.5]*3,
                       warmup_steps=20, first_schedule_index=0, checkpoint_policy='last')
        inputs[branch+'_spec'] = dict(population=hand,
            targets=np.searchsorted(known_ids, hand.labels) if branch == 'ood' else hand.labels,
            validation=known_validation,
            validation_targets=np.searchsorted(known_ids, known_validation.labels) if branch == 'ood' else known_validation.labels,
            branch=branch, options=options, initialization_path=checkpoint,
            output_directory=runtime/branch)
    for index, suffix in enumerate(('a', 'b')):
        font = data['fonts'][index]
        inputs['font_'+suffix+'_spec'] = dict(population=font,
            options=dict(epochs=60, batch_size=2, seed=3400+index*100, backend='sfc64'),
            output_directory=runtime/('font_'+suffix))
        inputs['gan_'+suffix+'_spec'] = dict(hand=hand, font=font,
            options=dict(epochs=40, batch_size=2, seed=3600+index*100, backend='sfc64', classifier_weight=4.),
            output_directory=runtime/('gan_'+suffix))
    return inputs


def independent_score(truth, predicted, weights):
    components = []
    for column in range(3):
        recalls = []
        for label in sorted(set(truth[weights > 0, column])):
            denominator = sum(float(w) for t, w in zip(truth[:, column], weights) if t == label)
            numerator = sum(float(w) for t, p, w in zip(truth[:, column], predicted[:, column], weights)
                            if t == label and p == label)
            recalls.append(numerator/denominator)
        components.append(sum(recalls)/len(recalls))
    return (2*components[0]+components[1]+components[2])/4


def main(checkpoint, runtime, output):
    baseline = prerequisites()
    paths = set(ROOT.glob('sciona/bengali_*.py')) | set(ROOT.glob('sciona/grapheme_*.py'))
    paths |= {ROOT/name for name in ('sciona/confidence_routing.py', 'sciona/component_recall.py',
        'sciona/svhn_autoaugment.py', 'scripts/grapheme_synthetic_domains.py',
        'scripts/review_video_numeric_runtime.py', 'sciona/services/execution_graph_codec.py',
        'sciona/visualizer/runner.py')}
    paths |= {Path(__file__).resolve(), PROVIDER}
    hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else '../sciona-atoms-dl/src/sciona/atoms/dl/grapheme_execution.py': sha(path)
              for path in sorted(paths)}
    runtime_checker.ROOT_PACKAGES = ['numpy', 'torch', 'efficientnet-pytorch', 'pillow', 'scipy',
                                     'opencv-python-headless', 'threadpoolctl']
    runtime_before = runtime_checker.review()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    if cv2.getNumThreads() != 1:
        raise ValueError('Single-thread OpenCV required')
    runner._ensure_atoms_imported()
    data = complete_populations()
    known = set(data['hand'].labels.tolist())
    unseen = set(data['fonts'][0].labels.tolist()) - known
    assert unseen and all(unseen <= set(font.labels.tolist()) for font in data['fonts'])
    assert unseen <= set(data['calibration'].labels.tolist()) & set(data['query'].labels.tolist())
    if output.exists():
        raise ValueError('Fresh output required')
    runtime.mkdir(parents=True, exist_ok=False)
    inputs = inputs_for(data, runtime, checkpoint)
    graph = build_grapheme_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    decoded = decode_execution_graph(nodes, edges, digest)
    assert decoded == graph
    values, handoffs = {}, set()
    incoming = {(edge.target_id, 'in_'+edge.input_name): (edge.source_id, 'out_'+edge.output_name) for edge in graph.edges}
    def capture(directory, node, name, value):
        key = (node, name)
        if key in incoming:
            if value is not values[incoming[key]]:
                raise ValueError('Graph handoff identity changed')
            handoffs.add(key)
        if name.startswith('out_'):
            values[key] = value
            print(json.dumps(dict(completed_node=node, output=name)), flush=True)
    with threadpool_limits(limits=1):
        with patch.object(runner, 'RUNS_DIR', runtime/'runner'), patch.object(runner, 'save_intermediate_value', side_effect=capture):
            result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-grapheme', 'complete').execute(inputs, cdg=decoded))
        assert result['status'] == 'completed' and len(result['trace']) == 22
        assert not any(row['cached'] for row in result['trace']) and len(handoffs) == 32
        training = {}
        for node, epochs, updates in [('train_ood', 200, 400), ('train_seen', 200, 400),
                ('train_font_a', 60, 180), ('train_font_b', 60, 180),
                ('train_generator_a', 40, 120), ('train_generator_b', 40, 120)]:
            state = values[node, 'out_state']
            assert state['metrics']['epochs'] == epochs and state['metrics']['updates'] == updates
            assert sha(state['path']) == state['sha256']
            training[node] = state['metrics']
        for population, images in [('validation', data['calibration'].images), ('query', data['query'].images)]:
            for branch in ('ood', 'seen', 'font_a', 'font_b'):
                generated = branch.startswith('font_')
                source = 'train_generator_'+branch[-1] if generated else 'train_'+branch
                replay = (predict_generated if generated else predict_b7)(values[source, 'out_state'], images)
                torch.testing.assert_close(replay, values[branch+'_'+population, 'out_logits'], rtol=0, atol=0)
        calibration = values['calibration', 'out_calibration']
        scores = values['scores_validation', 'out_scores']
        alternate = values['ensemble_validation', 'out_predictions']['joint_classes'].numpy()
        expected_scores = []
        for threshold in inputs['thresholds']:
            choose = (scores['ood_confidence'] < threshold) & ~inputs['known_class_mask'][alternate]
            joint = np.where(choose, alternate, scores['seen_joint_classes'])
            components = np.array([[int(v)//88, (int(v)//8)%11, 2 if int(v)%8 == 7 else int(v)%8] for v in joint])
            expected_scores.append(independent_score(inputs['truth_components'], components, inputs['sample_weight']))
        np.testing.assert_allclose(calibration['candidate_utilities'], expected_scores, rtol=1e-14, atol=1e-14)
        # Scores can differ by roundoff across equivalent reduction orders.
        best_score = max(expected_scores)
        assert abs(calibration['weighted_utility']-best_score) < 1e-14
        scores = values['scores_query', 'out_scores']
        alternate = values['ensemble_query', 'out_predictions']['joint_classes'].numpy()
        choose = (scores['ood_confidence'] < calibration['threshold']) & ~inputs['known_class_mask'][alternate]
        joint = np.where(choose, alternate, scores['seen_joint_classes'])
        expected = np.array([[int(v)//88, (int(v)//8)%11, 2 if int(v)%8 == 7 else int(v)%8] for v in joint])
        np.testing.assert_array_equal(values['submission', 'out_components'], expected)
        assert expected.shape == (3, 3)
        pools = threadpool_info()
        assert all(pool['num_threads'] == 1 for pool in pools)
    if prerequisites() != baseline or hashes != {name: sha(ROOT/name) for name in hashes}:
        raise ValueError('Bound implementation changed during execution')
    if runtime_checker.review() != runtime_before:
        raise ValueError('Installed runtime changed during execution')
    # Retain training-time digests privately for subsequent checkpoint handoffs.
    states = {node: values[node, 'out_state'] for node in training}
    (runtime/'checkpoint_receipt.json').write_text(json.dumps(dict(states=states, graph_sha256=digest,
        implementation_sha256=hashes, synthetic_only=True, approved=False), indent=2)+'\n')
    report = dict(format='bengali-complete-graph.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, graph_sha256=digest, actual_runner_nodes=22, all_nodes_uncached=True,
        exact_edge_handoffs=32, training=training, inference_checkpoint_replays=8,
        independent_calibration_scores_match=True, final_routing_and_components_match=True,
        known_training_classes=len(known), font_only_classes=len(unseen),
        calibration_examples=3, query_examples=3, query_alternate_routes=int(choose.sum()),
        fitted_threshold=calibration['threshold'], implementation_sha256=hashes,
        prerequisite_sha256=baseline, runtime=runtime_before,
        resource_policy=dict(compute_threads=1, interop_threads=1, opencv_threads=1, data_loader_workers=0),
        limits=['Constructed geometric symbols only; execution is not glyph-recognition, zero-shot accuracy or empirical transfer evidence.',
                'Published epoch counts with batch 2 and small synthetic populations; original training update counts are not reproduced.',
                'Explicit B7 normalization, warmup, shuffle, final checkpoint and seen decoding policies do not establish exact historical choices.',
                'Calibration weights and 21-point threshold grid are caller choices; calibration score is not an unbiased test score.',
                'Provisioned local filesystem and installed runtime only; source/reuse/provider/catalog publication reviews remain pending.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False, catalog_mutations=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.runtime, args.output)
