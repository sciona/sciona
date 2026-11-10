"""Full current-runtime Cornell graph qualification with bounded serial compute."""
import asyncio
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import sciona.atoms.ml.cornell_execution as provider
from sciona import cornell_lifecycle as lifecycle
from sciona.cornell_graph import build_cornell_graph
from sciona.cornell_augmentation import PINS, DEPENDENCIES, manifest
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts.cornell_synthetic import payload
from scripts.restore_cornell_reviewed_software import ROOT, sha, verify
from scripts import review_video_numeric_runtime as runtime_checker

EVIDENCE = 'docs/reviews/cornell_current_contracts.json'


def prerequisites():
    raw = (ROOT/EVIDENCE).read_bytes()
    if raw != subprocess.check_output(['git', 'show', '27068563:'+EVIDENCE], cwd=ROOT):
        raise ValueError('Immutable prerequisite evidence changed')
    report = json.loads(raw)
    if report['passed'] is not True or report['tests'] != dict(tests=31, failures=0, errors=0, skipped=0):
        raise ValueError('Passing contract evidence required')
    for path, digest in report['software_sha256'].items():
        if sha((ROOT/path).read_bytes()) != digest:
            raise ValueError('Qualified software changed')
    verify(Path('/private/tmp/sciona_cornell_source'), manifest('competition_cornell_source_pins.json', PINS)['pins'])
    for dep in manifest('competition_cornell_historical_dependencies.json', DEPENDENCIES):
        verify(Path('/private/tmp/sciona_cornell_dependencies')/dep['wheel'].split('-')[0], dep['pins'])
    previous = runtime_checker.ROOT_PACKAGES
    runtime_checker.ROOT_PACKAGES = report['runtime']['root_packages']
    try:
        if runtime_checker.review() != report['runtime']:
            raise ValueError('Qualified installed runtime changed')
    finally:
        runtime_checker.ROOT_PACKAGES = previous
    return report


def validate():
    baseline = prerequisites()
    paths = list(baseline['software_sha256']) + [
        'scripts/validate_cornell_current_graph.py', 'sciona/services/execution_graph_codec.py',
        'sciona/visualizer/runner.py', os.path.relpath(inspect.getfile(provider), ROOT)]
    before = {p:sha((ROOT/p).read_bytes()) for p in paths}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    digest, nodes, edges = encode_execution_graph(build_cornell_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph codec changed')
    captured = {}; phases = []
    original_phase = lifecycle.train_phase

    def observed_phase(*args, **kwargs):
        print(json.dumps(dict(phase=len(phases)+1, status='started', epochs=kwargs['epochs'])), flush=True)
        result = original_phase(*args, **kwargs)
        if len(result['reports']) != kwargs['epochs'] or result['selected_epoch'] != 2:
            raise ValueError('Complete configured training and checkpoint reload required')
        phases.append(dict(epochs=kwargs['epochs'], optimizer_steps=result['reports'][-1]['iterations'],
                           mixup=kwargs['mixup'], selected_epoch=result['selected_epoch']))
        print(json.dumps(dict(phase=len(phases), status='completed')), flush=True)
        return result

    def capture(directory, node, name, value):
        if (node, name) in [('prepare', 'out_prepared'), ('execute', 'in_prepared'), ('execute', 'out_result')]:
            captured[node, name] = value

    runner._ensure_atoms_imported()
    with tempfile.TemporaryDirectory(prefix='synthetic-cornell-current-') as temporary, threadpool_limits(limits=1):
        with patch.dict(os.environ, dict(SCIONA_CORNELL_SOURCE_DIR='/private/tmp/sciona_cornell_source',
                SCIONA_CORNELL_DEPENDENCY_DIR='/private/tmp/sciona_cornell_dependencies/audiomentations')), \
             patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture), \
             patch.object(lifecycle, 'train_phase', side_effect=observed_phase):
            result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-cornell', 'current').execute(
                {'payload':payload()}, cdg=graph))
    if result['status'] != 'completed':
        raise ValueError('Complete graph execution failed: '+str(result.get('error', result['status'])))
    if {t['node_id'] for t in result['trace']} != {'prepare', 'execute'} or any(t['cached'] for t in result['trace']):
        raise ValueError('All registered nodes must execute uncached')
    if captured['prepare', 'out_prepared'] is not captured['execute', 'in_prepared']:
        raise ValueError('Intermediate state routing changed')
    values = json.loads(json.dumps(captured['execute', 'out_result'], allow_nan=False))
    if (values['models_completed'], len(values['training_phases']), len(phases)) != (13, 17, 17):
        raise ValueError('Complete model and phase inventory required')
    if sum(p['phase']=='continuation' for p in values['training_phases']) != 4 or sum(p['mixup'] for p in phases) != 8:
        raise ValueError('Four full mixup members and continuations required')
    if {p['member'] for p in values['training_phases']} != set(range(13)):
        raise ValueError('Missing ensemble member')
    votes = np.asarray(values['window_votes'])
    if votes.shape != (6,264) or len(values['whole_record_decisions']) != 264 or np.any((votes<0)|(votes>13)):
        raise ValueError('Complete finite voting output required')
    if before != {p:sha((ROOT/p).read_bytes()) for p in paths} or prerequisites() != baseline:
        raise ValueError('Software changed during execution')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        graph_sha256=digest, software_sha256=before, prerequisite_sha256=sha((ROOT/EVIDENCE).read_bytes()),
        models_completed=13, training_phases=17, continuations=4, phases=phases,
        actual_runner_nodes=2, all_nodes_uncached=True, exact_state_routing=True,
        full_thirty_second_training=True, ten_copy_inference=True, strict_json_output=True,
        resource_policy=dict(serial_models=True, torch_threads=1, interop_threads=1, native_threads=1),
        limits=['All configured synthetic phases use two epochs and explicit random initialization; no historical pretrained accuracy claim.',
                'Existing source branches, full model topology, checkpoint selection and voting execute unchanged.',
                'Original source corrections, reusable provider contracts and catalog publication remain separate gates.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/cornell_current_graph.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, models=13, phases=17, approved=False)), flush=True)
