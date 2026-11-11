"""Execute all 220 M5 models and compare the complete reference lifecycle."""
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
from scripts.validate_m5_full_inventory import configuration
from sciona.m5_execution import prepare,execute
from sciona.decomposed_m5_graph import build_decomposed_m5_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload():
    config=json.loads(json.dumps(configuration(),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v))
    return dict(version=1,identities=['synthetic-'+str(i) for i in range(70)],configuration=config,
                controls=dict(recursive_first_day=0,nonrecursive_first_day=710))


def validate():
    graph=build_decomposed_m5_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    graph=decode_execution_graph(nodes,edges,digest)
    if encode_execution_graph(graph)[0]!=digest:raise ValueError('Graph codec mismatch')
    paths={str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('m5_*.py')}
    paths|={'sciona/decomposed_m5_graph.py','sciona/decomposed_m5_runtime.py',
            'scripts/validate_m5_full_inventory.py','scripts/validate_decomposed_m5_graph_execution.py',
            'sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py'}
    before={name:sha(ROOT/name) for name in sorted(paths)}
    providers={n.matched_primitive:Path(inspect.getfile(importlib.import_module(n.matched_primitive.rsplit('.',1)[0]))) for n in graph.nodes}
    provider_before={name:sha(path) for name,path in providers.items()}
    packages=[line.split('==')[0] for line in (ROOT/'requirements/m5-execution.txt').read_text().splitlines() if line]
    versions={name:metadata.version(name) for name in packages}
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if name.startswith('out_'):captured[node,name]=value
    inputs=payload()
    print('Executing all 220 models and complete 28-day graph',flush=True)
    with tempfile.TemporaryDirectory(prefix='m5-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-m5','case').execute({'payload':inputs},cdg=graph))
    if status['status']!='completed' or len(status['trace'])!=7 or any(t['cached'] for t in status['trace']):
        raise ValueError('All seven graph nodes must execute uncached')
    if set(captured)!={(n.node_id,'out_'+p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('Incomplete actual graph outputs')
    result=json.loads(json.dumps(captured['result','out_result'],allow_nan=False))
    models=captured['fit','out_fitted']['models']
    counts=[sum((m.recursive,m.pooling)==(recursive,pooling) for m in models)
            for recursive in (True,False) for pooling in ('outlet','outlet_category','outlet_department')]
    if len(models)!=220 or counts!=[10,30,70,10,30,70] or any(m.report['evaluated_rounds']!=3000 for m in models):
        raise ValueError('Full family inventory and training schedules required')
    if (np.asarray(result['forecast']).shape!=(70,28) or result['models']!=220
            or result['horizon']!=28 or result['model_families']!=6):
        raise ValueError('Complete forecast output required')
    print('Graph completed; retraining full reference lifecycle for exact output comparison',flush=True)
    if result!=execute(prepare(inputs)):raise ValueError('Complete reference result differs')
    if before!={name:sha(ROOT/name) for name in paths} or provider_before!={name:sha(path) for name,path in providers.items()}:
        raise ValueError('Implementation changed during execution')
    if versions!={name:metadata.version(name) for name in versions}:raise ValueError('Runtime changed')
    return dict(format='decomposed-m5-execution.v1',status='passed',approved=False,catalog_mutations=0,
        synthetic_only=True,all_nodes_uncached=True,serialized_graph_sha256=digest,
        code_sha256=before,bound_provider_sha256=provider_before,runtime=versions,
        resource_policy=dict(native_threads=1,lightgbm_threads=1,serial_models=True),
        checks=dict(actual_runner_nodes=7,actual_runner_edges=11,models_fitted=220,
            family_model_counts=counts,configured_rounds_per_model=3000,evaluated_rounds_total=660000,
            synthetic_series=70,horizon=28,exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True),
        limits=['Full synthetic reconstruction parity does not establish historical competition accuracy or sample-volume equivalence.',
                'Current package closure, provider semantics and original-intake publication remain to be qualified.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/decomposed_m5_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)
