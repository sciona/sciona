"""Execute all 126 uncertainty models and compare the complete reference lifecycle."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import asyncio
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from scripts.validate_m5u_graph_execution import payload
from sciona.m5u_training import PARAMETERS,QUANTILES,schedule
from sciona.m5u_execution import prepare,execute
from sciona.decomposed_m5u_graph import build_decomposed_m5u_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    graph=build_decomposed_m5u_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    graph=decode_execution_graph(nodes,edges,digest)
    if encode_execution_graph(graph)[0]!=digest:raise ValueError('Graph codec mismatch')
    paths={str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('m5u_*.py')}
    paths|={'sciona/decomposed_m5u_graph.py','sciona/decomposed_m5u_runtime.py',
            'scripts/validate_m5u_graph_execution.py','sciona/m5u_search_parameters.json','scripts/validate_decomposed_m5u_graph_execution.py',
            'sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py'}
    before={name:sha(ROOT/name) for name in sorted(paths)}
    providers={n.matched_primitive:Path(inspect.getfile(importlib.import_module(n.matched_primitive.rsplit('.',1)[0]))) for n in graph.nodes}
    provider_before={name:sha(path) for name,path in providers.items()}
    packages=[line.split('==')[0] for line in (ROOT/'requirements/m5u-execution.txt').read_text().splitlines() if line]
    versions={name:metadata.version(name) for name in packages}
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if name.startswith('out_'):captured[node,name]=value
    inputs=payload()
    print('Executing all 126 models and complete 28-day graph',flush=True)
    with tempfile.TemporaryDirectory(prefix='m5u-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-m5u','case').execute({'payload':inputs},cdg=graph))
    if status['status']!='completed' or len(status['trace'])!=6 or any(t['cached'] for t in status['trace']):
        raise ValueError('All six graph nodes must execute uncached')
    if set(captured)!={(n.node_id,'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Incomplete actual graph outputs')
    result=json.loads(json.dumps(captured['result','out_result'],allow_nan=False))
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
    print('Graph completed; retraining full reference lifecycle for exact output comparison',flush=True)
    reference=json.loads(json.dumps(execute(prepare(inputs)),allow_nan=False))
    if result!=reference:raise ValueError('Complete reference result differs')
    if before!={name:sha(ROOT/name) for name in paths} or provider_before!={name:sha(path) for name,path in providers.items()}:
        raise ValueError('Implementation changed during execution')
    if versions!={name:metadata.version(name) for name in versions}:raise ValueError('Runtime changed')
    return dict(format='decomposed-m5u-execution.v1',status='passed',approved=False,catalog_mutations=0,
        synthetic_only=True,all_nodes_uncached=True,serialized_graph_sha256=digest,
        code_sha256=before,bound_provider_sha256=provider_before,runtime=versions,
        resource_policy=dict(native_threads=1,lightgbm_threads=1,search_workers=1,serial_models=True),
        checks=dict(actual_runner_nodes=6,actual_runner_edges=8,models_fitted=126,
            levels=14,quantiles=9,horizon=28,hierarchy_series=39,candidate_fits=4482,refits=126,
            query_rows=2730000,full_source_sampling_and_search=True,outer_group_isolation=True,
            exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True),
        limits=['Full synthetic reconstruction parity does not establish historical competition accuracy or sample-volume equivalence.',
                'Current package closure, provider semantics and original-intake publication remain to be qualified.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/decomposed_m5u_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)
