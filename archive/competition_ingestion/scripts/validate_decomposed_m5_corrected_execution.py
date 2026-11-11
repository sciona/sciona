"""Execute the stored corrected M5 original with generated inputs and the full 220-model lifecycle."""
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

import numpy as np
from scripts.validate_decomposed_m5_graph_execution import payload
from sciona.m5_execution import prepare, execute
from scripts import stage_m5_corrected_intake as intake
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
    baseline = json.loads((ROOT/'docs/reviews/decomposed_m5_graph_execution.json').read_text())
    paths = set(baseline['code_sha256'])
    paths |= {'scripts/validate_decomposed_m5_corrected_execution.py',
              'scripts/stage_m5_corrected_intake.py','scripts/decomposed_m5_approved_provider_state.py',
              'docs/reviews/m5_corrected_intake_plan.json','docs/reviews/m5_approved_provider_state.json'}
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
    print('Executing full stored 220-model candidate', flush=True)
    with tempfile.TemporaryDirectory(prefix='m5-synthetic-graph-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-m5', 'case').execute(
                {'payload': inputs}, cdg=graph))
    if status['status'] != 'completed':
        raise ValueError('Full graph execution failed')
    if len(status['trace']) != 7 or any(t['cached'] for t in status['trace']):
        raise ValueError('Every stored node must execute uncached')
    if set(captured) != {(n.node_id, 'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Incomplete actual graph outputs')
    result = json.loads(json.dumps(captured['result','out_result'], allow_nan=False))
    print('Stored execution complete; comparing full reference lifecycle', flush=True)
    if result != execute(prepare(inputs)):
        raise ValueError('Complete reconstruction result differs')
    models = captured['fit','out_fitted']['models']
    counts = [sum((m.recursive,m.pooling)==(recursive,pooling) for m in models)
              for recursive in (True,False) for pooling in ('outlet','outlet_category','outlet_department')]
    if len(models)!=220 or counts!=[10,30,70,10,30,70] or any(m.report['evaluated_rounds']!=3000 for m in models):
        raise ValueError('Full family inventory and training schedules required')
    if (np.asarray(result['forecast']).shape!=(70,28) or result['models']!=220
            or result['horizon']!=28 or result['model_families']!=6):
        raise ValueError('Complete forecast output required')
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
    return dict(format='m5-corrected-execution-validation.v1', status='passed', approved=False,
        catalog_mutations=0, synthetic_only=True, serialized_graph_sha256=digest,
        stored_candidate=True, all_nodes_uncached=True, candidate_version_id=plan['version_id'],
        plan_sha256=intake.sha(intake.PLAN),
        checks=dict(actual_runner_nodes=7,actual_runner_edges=11,models_fitted=220,
            family_model_counts=counts,configured_rounds_per_model=3000,evaluated_rounds_total=660000,
            synthetic_series=70,horizon=28,
            exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True),
        code_sha256=before, bound_provider_sha256=provider_before, runtime=runtime_before,
        resource_policy=dict(native_threads=1, lightgbm_threads=1, serial_models=True),
        limits=['Exact comparison is against the existing full six-family forecast reconstruction; historical winning-agent equivalence is unproven.',
                'Synthetic computation does not establish generalization, historical training-population equivalence, clean installation or catalog eligibility.',
                'All seven exact approved providers are checked before and after execution. Original-intake publication remains pending.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/m5_corrected_intake_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))
