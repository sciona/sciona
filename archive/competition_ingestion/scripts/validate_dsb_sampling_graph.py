"""Run the unchanged independent source oracle on the draft sampling graph."""
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
from sciona.dsb_decomposed_execution import build_dsb_sampling_graph
from sciona.atoms.numerical.statistics import threshold_sampling
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
    calls = dict(repetition=0, selection=0)
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
    with patch.object(oracle, 'build_dsb_raw_training_graph', build_dsb_sampling_graph), \
         patch.object(oracle.runner.CDGExecutionSession, 'execute', execute), \
         patch.object(sampling, 'repeat_rows_by_threshold', repeat_spy), \
         patch.object(sampling, 'sequential_softmax_indices', select_spy):
        result = await oracle.validate(ROOT, source_root, include_inference=True)
    if sha(Path(oracle.__file__)) != oracle_hash:
        raise ValueError('Oracle changed during execution')
    retained = json.loads((ROOT/'docs/reviews/competition_dsb_full_source_execution.json').read_text())
    if result['cases'] != retained['cases'] or len(executions) != 2:
        raise ValueError('Complete retained training/inference scope required')
    result.update(passed=True, catalog_mutations=0, executions=executions,
        resource_policy=dict(serial=True,native_threads=1,torch_interop_threads=1),
        numerical_provider_sha256=sha(Path(inspect.getfile(threshold_sampling))),
        validator_sha256=sha(Path(__file__)),oracle_sha256=oracle_hash,
        retained_source_evidence_sha256=sha(ROOT/'docs/reviews/competition_dsb_full_source_execution.json'))
    result['limitations'] += [
        'Intermediate graph: sampling is reusable; remaining preparation, inference, loss and runtime decomposition is pending.',
        'Two three-epoch windows are qualified; full competition training is not claimed.',
        'This result is not catalog publication approval.',
    ]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(args.source_root))
    (ROOT/'docs/reviews/dsb_sampling_graph_source_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','approved','cases','executions')}),flush=True)
