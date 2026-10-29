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
from sciona.dsb_decomposed_loss_graph import build_dsb_loss_graph
from sciona import dsb_decomposed_losses as losses
from sciona.atoms.dl.loss import thresholded_log
from sciona import dsb_decomposed_network as network
from sciona.atoms.dl import probability_aggregation
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
    calls = dict(repetition=0, selection=0, pooling_with_gradients=0, pooling_without_gradients=0, penalty_with_gradients=0, penalty_without_gradients=0)
    penalty = losses.thresholded_log_penalty
    def penalty_spy(*args, **kwargs):
        calls['penalty_with_gradients' if torch.is_grad_enabled() else 'penalty_without_gradients'] += 1
        return penalty(*args, **kwargs)
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
            raise ValueError('Graph execution must call both reusable operations')
        executions.append(dict(uncached_nodes=len(expected), reusable_calls=used))
        return result
    oracle_hash = sha(Path(oracle.__file__))
    with patch.object(oracle, 'build_dsb_raw_training_graph', build_dsb_loss_graph), \
         patch.object(oracle.runner.CDGExecutionSession, 'execute', execute), \
         patch.object(sampling, 'repeat_rows_by_threshold', repeat_spy), \
         patch.object(sampling, 'sequential_softmax_indices', select_spy), \
         patch.object(network, 'noisy_or_with_baseline', pooling_spy), \
         patch.object(losses, 'thresholded_log_penalty', penalty_spy):
        result = await oracle.validate(ROOT, source_root, include_inference=True)
    if sha(Path(oracle.__file__)) != oracle_hash:
        raise ValueError('Oracle changed during execution')
    retained = json.loads((ROOT/'docs/reviews/competition_dsb_full_source_execution.json').read_text())
    expected_cases=[dict(case,executed_nodes=len(build_dsb_loss_graph().nodes)) for case in retained['cases']]
    if result['cases'] != expected_cases or len(executions) != 2:
        raise ValueError('Complete retained training/inference scope required')
    result.update(passed=True, catalog_mutations=0, executions=executions,
        resource_policy=dict(serial=True,native_threads=1,torch_interop_threads=1),
        numerical_provider_sha256=sha(Path(inspect.getfile(threshold_sampling))),
        tiling_provider_sha256=sha(Path(inspect.getfile(volume_tiles))),
        pooling_provider_sha256=sha(Path(inspect.getfile(probability_aggregation))),
        penalty_provider_sha256=sha(Path(inspect.getfile(thresholded_log))),
        validator_sha256=sha(Path(__file__)),oracle_sha256=oracle_hash,
        retained_source_evidence_sha256=sha(ROOT/'docs/reviews/competition_dsb_full_source_execution.json'))
    result['limitations'] += [
        'Intermediate graph: sampling and tiling are reusable; preparation, classifier internals, losses and runtime decomposition remain.',
        'Two three-epoch windows are qualified; full competition training is not claimed.',
        'This result is not catalog publication approval.',
    ]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(args.source_root))
    (ROOT/'docs/reviews/dsb_loss_graph_source_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','approved','cases','executions')}),flush=True)
