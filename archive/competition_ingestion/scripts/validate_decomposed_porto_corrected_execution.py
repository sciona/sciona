"""Execute the stored corrected Porto original with generated inputs and the full six-model lifecycle."""
import asyncio
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from scripts.porto_synthetic import payload
from sciona.porto_execution import prepare, execute
from scripts import stage_porto_corrected_intake as intake
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    inputs = payload()
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        stored = intake.check_candidate(db, plan, providers)
    digest, nodes, edges = encode_execution_graph(stored)
    baseline = json.loads((ROOT/'docs/reviews/decomposed_porto_graph_execution.json').read_text())
    paths = set(baseline['code_sha256']) - {'scripts/validate_decomposed_porto_graph_execution.py'}
    paths |= {'scripts/validate_decomposed_porto_corrected_execution.py',
              'scripts/stage_porto_corrected_intake.py','scripts/decomposed_porto_approved_provider_state.py',
              'docs/reviews/porto_corrected_intake_plan.json','docs/reviews/porto_approved_provider_state.json'}
    before = {name:sha(ROOT/name) for name in sorted(paths)}
    provider_paths = {n.matched_primitive:Path(inspect.getfile(importlib.import_module(
        n.matched_primitive.rsplit('.',1)[0]))) for n in stored.nodes}
    provider_before = {name:sha(path) for name,path in provider_paths.items()}
    runtime_before = {name:metadata.version(name) for name in baseline['runtime']}
    if provider_before != baseline['bound_provider_sha256'] or runtime_before != baseline['runtime']:
        raise ValueError('Qualified providers or runtime changed')
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph codec mismatch')
    for node in graph.nodes:
        importlib.import_module(node.matched_primitive.rsplit('.',1)[0])
    runner._ensure_atoms_imported()
    captured = {}
    def capture(directory, node, name, value):
        if name.startswith('out_'):
            captured[node, name] = value
    print('Executing full stored six-model candidate', flush=True)
    with tempfile.TemporaryDirectory(prefix='porto-synthetic-graph-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-porto', 'case').execute(
                {'payload': inputs}, cdg=graph))
    if status['status'] != 'completed':
        raise ValueError('Full graph execution failed')
    if len(status['trace']) != 6 or any(t['cached'] for t in status['trace']):
        raise ValueError('Every stored node must execute uncached')
    if set(captured) != {(n.node_id, 'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Incomplete actual graph outputs')
    result = json.loads(json.dumps(captured['result','out_result'], allow_nan=False))
    print('Stored execution complete; comparing full reference lifecycle', flush=True)
    if result != execute(prepare(inputs)):
        raise ValueError('Complete reconstruction result differs')
    if result['models'] != 6 or result['dae_models'] != 5 or len(result['probabilities']) != 8:
        raise ValueError('Complete six-model and five-DAE lifecycle required')
    if len(result['learned_widths']) != 5 or result['score_kind'] != 'equal_probability_mean':
        raise ValueError('Complete learned-feature and output contracts required')
    fitted = captured['fit','out_fitted']
    if any(r['dae_epochs'] != 60 or len(r['training_loss']) != 60 for r in fitted['neural_records']):
        raise ValueError('Complete synthetic DAE and classifier schedules required')
    if before != {name:sha(ROOT/name) for name in paths}:
        raise ValueError('Execution implementation changed during run')
    if provider_before != {name:sha(path) for name,path in provider_paths.items()}:
        raise ValueError('Provider source changed during run')
    if runtime_before != {name:metadata.version(name) for name in runtime_before}:
        raise ValueError('Runtime changed during run')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if intake.check_candidate(db, plan, providers) != stored:
            raise ValueError('Stored candidate changed during execution')
    return dict(format='porto-corrected-execution-validation.v1', status='passed', approved=False,
        catalog_mutations=0, synthetic_only=True, serialized_graph_sha256=digest,
        stored_candidate=True, all_nodes_uncached=True, candidate_version_id=plan['version_id'],
        plan_sha256=intake.sha(intake.PLAN),
        checks=dict(actual_runner_nodes=6,actual_runner_edges=9,models_fitted=6,dae_models=5,
            synthetic_query_rows=8,full_configured_schedules=True,
            exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True),
        code_sha256=before, bound_provider_sha256=provider_before, runtime=runtime_before,
        resource_policy=dict(native_threads=1, lightgbm_threads=1, torch_threads=1, torch_interop_threads=1, serial_models=True),
        limits=['Exact comparison is against the existing full five-DAE/classifier and tree reconstruction; historical winning-agent equivalence is unproven.',
                'Synthetic computation does not establish generalization, historical training-population equivalence, clean installation or catalog eligibility.',
                'All six exact approved providers are checked before and after execution. Original-intake publication remains pending.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/porto_corrected_intake_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))
