"""Run the unchanged independent source oracle on the draft reusable-loss graph."""
import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
from sciona import dsb_decomposed_sampling as sampling
from sciona.dsb_decomposed_preparation_graph import build_dsb_preparation_graph
from sciona import dsb_decomposed_preparation as preparation
from sciona.atoms.numerical.spatial import resampling
from sciona import dsb_decomposed_losses as losses
from sciona.atoms.dl.loss import thresholded_log, ranked_selection
from sciona import dsb_decomposed_network as network
from sciona.atoms.dl import probability_aggregation
from sciona.atoms.dl.spatial import centered_window
from sciona.atoms.numerical.statistics import threshold_sampling
from sciona.atoms.numerical.spatial import volume_tiles
from scripts import validate_dsb_serialized_training as oracle

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def validate(source_root):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('Explicit single-thread environment required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original_execute = oracle.runner.CDGExecutionSession.execute
    repeat, select = sampling.repeat_rows_by_threshold, sampling.sequential_softmax_indices
    executions = []
    calls = dict(center_with_gradients=0, center_without_gradients=0, resampling=0, ranking=0, repetition=0, selection=0, pooling_with_gradients=0, pooling_without_gradients=0, penalty_with_gradients=0, penalty_without_gradients=0)
    resample = preparation.resample_regular_volume
    def resample_spy(*args, **kwargs):
        calls["resampling"] += 1
        return resample(*args, **kwargs)
    ranking = losses.highest_scoring_indices
    def ranking_spy(*args, **kwargs):
        if not torch.is_grad_enabled():
            raise ValueError("Validation must bypass hard mining")
        calls["ranking"] += 1
        return ranking(*args, **kwargs)
    penalty = losses.thresholded_log_penalty
    def penalty_spy(*args, **kwargs):
        calls['penalty_with_gradients' if torch.is_grad_enabled() else 'penalty_without_gradients'] += 1
        return penalty(*args, **kwargs)
    center = network.centered_window
    def center_spy(*args, **kwargs):
        calls["center_with_gradients" if torch.is_grad_enabled() else "center_without_gradients"] += 1
        return center(*args, **kwargs)
    pooling = network.noisy_or_with_baseline
    def pooling_spy(*args, **kwargs):
        calls['pooling_with_gradients' if torch.is_grad_enabled() else 'pooling_without_gradients'] += 1
        return pooling(*args, **kwargs)
    def repeat_spy(*args, **kwargs):
        calls['repetition'] += 1
        return repeat(*args, **kwargs)
    def select_spy(*args, **kwargs):
        calls['selection'] += 1
        return select(*args, **kwargs)
    async def execute(session, *args, **kwargs):
        before = dict(calls)
        result = await original_execute(session, *args, **kwargs)
        graph = kwargs['cdg']
        expected = {n.node_id for n in graph.nodes}
        if (result['status'] != 'completed' or len(result['trace']) != len(expected)
                or {t['node_id'] for t in result['trace']} != expected
                or any(t['cached'] for t in result['trace'])):
            raise ValueError('Complete uncached graph execution required')
        used = {k:calls[k]-before[k] for k in calls}
        if not all(used.values()):
            raise ValueError('Graph execution must call every instrumented reusable operation')
        executions.append(dict(uncached_nodes=len(expected), reusable_calls=used))
        return result
    oracle_hash = sha(Path(oracle.__file__))
    with patch.object(oracle, 'build_dsb_raw_training_graph', build_dsb_preparation_graph), \
         patch.object(oracle.runner.CDGExecutionSession, 'execute', execute), \
         patch.object(sampling, 'repeat_rows_by_threshold', repeat_spy), \
         patch.object(sampling, 'sequential_softmax_indices', select_spy), \
         patch.object(network, 'noisy_or_with_baseline', pooling_spy), \
         patch.object(losses, 'thresholded_log_penalty', penalty_spy), \
         patch.object(losses, 'highest_scoring_indices', ranking_spy), \
         patch.object(preparation, 'resample_regular_volume', resample_spy), \
         patch.object(network, 'centered_window', center_spy):
        result = await oracle.validate(ROOT, source_root, include_inference=True)
    if sha(Path(oracle.__file__)) != oracle_hash:
        raise ValueError('Oracle changed during execution')
    retained = json.loads((ROOT/'docs/reviews/competition_dsb_full_source_execution.json').read_text())
    expected_cases=[dict(case,executed_nodes=len(build_dsb_preparation_graph().nodes)) for case in retained['cases']]
    if result['cases'] != expected_cases or len(executions) != 2:
        raise ValueError('Complete retained training/inference scope required')
    result.update(passed=True, catalog_mutations=0, executions=executions,
        resource_policy=dict(serial=True,native_threads=1,torch_interop_threads=1),
        numerical_provider_sha256=sha(Path(inspect.getfile(threshold_sampling))),
        tiling_provider_sha256=sha(Path(inspect.getfile(volume_tiles))),
        pooling_provider_sha256=sha(Path(inspect.getfile(probability_aggregation))),
        centered_window_provider_sha256=sha(Path(inspect.getfile(centered_window))),
        resampling_provider_sha256=sha(Path(inspect.getfile(resampling))),
        ranking_provider_sha256=sha(Path(inspect.getfile(ranked_selection))),
        penalty_provider_sha256=sha(Path(inspect.getfile(thresholded_log))),
        validator_sha256=sha(Path(__file__)),oracle_sha256=oracle_hash,
        retained_source_evidence_sha256=sha(ROOT/'docs/reviews/competition_dsb_full_source_execution.json'))
    result['limitations'] += [
        'Reusable sampling, tiling, resampling, center windows, pooling and loss operations are integrated; coordinate, augmentation/anchor and lifecycle contract work remains.',
        'Two three-epoch windows are qualified; full competition training is not claimed.',
        'This result is not catalog publication approval.',
    ]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(args.source_root))
    (ROOT/'docs/reviews/dsb_center_graph_source_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','approved','cases','executions')}),flush=True)
