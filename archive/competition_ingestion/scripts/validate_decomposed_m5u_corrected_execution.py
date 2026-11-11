"""Execute the stored corrected M5 Uncertainty original with generated inputs and the full 126-model lifecycle."""
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
from scripts.validate_decomposed_m5u_graph_execution import payload
from sciona.m5u_execution import prepare, execute
from sciona.m5u_training import PARAMETERS, QUANTILES, schedule
from scripts import stage_m5u_corrected_intake as intake
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
    baseline = json.loads((ROOT/'docs/reviews/decomposed_m5u_graph_execution.json').read_text())
    paths = set(baseline['code_sha256'])
    paths |= {'scripts/validate_decomposed_m5u_corrected_execution.py',
              'scripts/stage_m5u_corrected_intake.py','scripts/decomposed_m5u_approved_provider_state.py',
              'docs/reviews/m5u_corrected_intake_plan.json','docs/reviews/m5u_approved_provider_state.json'}
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
    print('Executing full stored 126-model candidate', flush=True)
    with tempfile.TemporaryDirectory(prefix='m5u-synthetic-graph-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-m5u', 'case').execute(
                {'payload': inputs}, cdg=graph))
    if status['status'] != 'completed':
        raise ValueError('Full graph execution failed')
    if len(status['trace']) != 6 or any(t['cached'] for t in status['trace']):
        raise ValueError('Every stored node must execute uncached')
    if set(captured) != {(n.node_id, 'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Incomplete actual graph outputs')
    result = json.loads(json.dumps(captured['result','out_result'], allow_nan=False))
    print('Stored execution complete; comparing full reference lifecycle', flush=True)
    if result != json.loads(json.dumps(execute(prepare(inputs)), allow_nan=False)):
        raise ValueError('Complete reconstruction result differs')
    raw=captured['fit_forecast','out_forecasted']
    reports=raw['reports']
    if (set(reports)!=set(PARAMETERS) or len(reports)!=14
            or sum(r['models'] for r in reports.values())!=126
            or sum(r['candidate_fits'] for r in reports.values())!=4482
            or sum(r['refits'] for r in reports.values())!=126
            or sum(r['query_rows'] for r in reports.values())!=2730000
            or not all(r['outer_group_isolation'] for r in reports.values())):
        raise ValueError('Full level inventory, search schedule and repeated query budgets required')
    for level,report in reports.items():
        factors=np.unique(raw['factors'][raw['levels']==level])
        if factors.size!=1:raise ValueError('Unique level factor required')
        expected=schedule(int(level),float(factors[0]))
        if any(report[k]!=v for k,v in expected.items()) or report['models']!=9:
            raise ValueError('Full source sampling and search controls required')
    predictions=np.asarray(result['forecast'])
    if (predictions.shape!=(9,28,39) or not np.isfinite(predictions).all()
            or result['hierarchy_indices']!=list(range(39)) or result['horizon']!=28
            or result['quantiles']!=list(QUANTILES) or result['normalization']!='retained_base_reference'):
        raise ValueError('Complete hierarchy, quantile and horizon output required')
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
    return dict(format='m5u-corrected-execution-validation.v1', status='passed', approved=False,
        catalog_mutations=0, synthetic_only=True, serialized_graph_sha256=digest,
        stored_candidate=True, all_nodes_uncached=True, candidate_version_id=plan['version_id'],
        plan_sha256=intake.sha(intake.PLAN),
        checks=dict(actual_runner_nodes=6,actual_runner_edges=8,models_fitted=126,
            levels=14,quantiles=9,horizon=28,hierarchy_series=39,candidate_fits=4482,refits=126,
            query_rows=2730000,full_source_sampling_and_search=True,outer_group_isolation=True,
            exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True),
        code_sha256=before, bound_provider_sha256=provider_before, runtime=runtime_before,
        resource_policy=dict(native_threads=1, lightgbm_threads=1, search_workers=1, serial_models=True),
        limits=['Exact comparison is against the existing full direct-quantile reconstruction; historical winning-agent equivalence is unproven.',
                'Synthetic computation does not establish generalization, historical training-population equivalence, clean installation or catalog eligibility.',
                'All six exact approved providers are checked before and after execution. Original-intake publication remains pending.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/m5u_corrected_intake_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))
