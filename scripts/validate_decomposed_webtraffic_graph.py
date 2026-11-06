"""Execute complete registered graph and verify source-reference final records."""
import asyncio
import copy
import importlib
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_info
import sciona.atoms.dl.decomposed_webtraffic_execution as providers
from sciona.decomposed_webtraffic_graph import build_decomposed_webtraffic_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
from sciona import decomposed_webtraffic_lifecycle as lifecycle
from sciona.webtraffic_payload import execute_payload
from scripts.validate_webtraffic_graph import synthetic_payload
from scripts.audit_decomposed_webtraffic_readiness import ROOT,sha


def validate():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    paths=['sciona/decomposed_webtraffic_'+s+'.py' for s in ('contract','features','lifecycle','graph')]
    paths+=['sciona/panel_output.py','scripts/validate_decomposed_webtraffic_graph.py','sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('webtraffic_*.py'))]
    before={p:sha(ROOT/p) for p in paths}
    graph=build_decomposed_webtraffic_graph();digest,nodes,edges=encode_execution_graph(graph)
    graph=decode_execution_graph(nodes,edges,digest)
    if encode_execution_graph(graph)[0]!=digest or len(nodes)!=19:raise ValueError('Complete graph roundtrip required')
    hashes={}
    for node in graph.nodes:
        fn=getattr(providers,node.matched_primitive.rsplit('.',1)[-1])
        if list(inspect.signature(fn).parameters)!=[p.name for p in node.inputs]:raise ValueError('Callable inputs differ')
        hashes[node.matched_primitive]=sha(inspect.getfile(inspect.unwrap(fn)))
    runner._ensure_atoms_imported();cases=[];original_train=lifecycle.train
    for seed,phase,steps,width in [(2,'after',1,3),(7,'before',2,63)]:
        reference=synthetic_payload()
        reference['config'].update(seed=seed,max_steps=steps,ema_parameter_phase=phase,ema_step_phase=phase)
        payload=copy.deepcopy(reference);payload['version']=2
        dates=[d.date().isoformat() for d in pd.date_range(pd.Timestamp(payload['start_day'])+pd.Timedelta(days=800),periods=63)]
        ids=np.arange(len(payload['pages'])*width,dtype=np.int64).reshape(-1,width)
        payload['output']=dict(rows=payload['pages'].copy(),columns=dates[-width:],identifiers=ids.tolist())
        saved=copy.deepcopy(payload);calls=[];checkpoints=[];captured={}
        def tracked_train(prepared,initialized,advance_function,checkpoint_function):
            if advance_function is not providers.advance or checkpoint_function is not providers.save_checkpoint:
                raise ValueError('Exact registered callable bindings required')
            def update(state,batches,observations):
                calls.append(state['shared_step']);return advance_function(state,batches,observations)
            def retain(checkpoints_,state):
                checkpoints.append(state['shared_step']);return checkpoint_function(checkpoints_,state)
            return original_train(prepared,initialized,update,retain)
        def capture(directory,node,name,value):
            if name.startswith('out_'):captured[node,name]=value
        with tempfile.TemporaryDirectory(prefix='synthetic-webtraffic-decomposed-') as directory:
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture),patch.object(lifecycle,'train',side_effect=tracked_train):
                status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-webtraffic','decomposed').execute({'payload':payload},cdg=graph))
        if status['status']!='completed':raise ValueError('Complete graph failed')
        traces={t['node_id']:t for t in status['trace']}
        if set(traces)!={n.node_id for n in graph.nodes} or any(t['cached'] for t in traces.values()):raise ValueError('Every graph node must execute uncached')
        if any(traces[n].get('callable_reference') is not True for n in ('advance','save_checkpoint')):
            raise ValueError('Explicit callable nodes required')
        expected=execute_payload(reference)
        result=dict(version=2,identifiers=ids.ravel().tolist(),values=[v for row in expected['forecasts'] for v in row[-width:]])
        if captured['result','out_result']!=result or payload!=saved:raise ValueError('Complete reference result or input preservation differs')
        if calls!=list(range(steps+1)) or checkpoints!=list(range(1,steps+2)):
            raise ValueError('All configured updates and saved checkpoints required')
        cases.append(dict(seed=seed,ema_phase=phase,configured_max_steps=steps,actual_updates=len(calls),
            saved_checkpoints=len(checkpoints),output_columns=width,output_rows=len(payload['pages']),
            full_result_equal=True,all_nodes_uncached=True,exact_registered_callables=True))
    if any(p['num_threads']!=1 for p in threadpool_info()):raise ValueError('Native thread budget exceeded')
    if any(sha(inspect.getfile(providers))!=h for h in hashes.values()):raise ValueError('Provider changed during execution')
    if before!={p:sha(ROOT/p) for p in paths}:raise ValueError('Implementation changed during execution')
    return dict(format='decomposed-webtraffic-graph.v1',passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        graph_sha256=digest,nodes=len(nodes),edges=len(edges),distinct_providers=len(hashes),cases=cases,
        provider_sha256=hashes,implementation_sha256=before,
        resource_policy=dict(serial_cases=True,native_threads=1,torch_interop_threads=1),
        limits=['Registered local execution only; exact catalog intake, provider publication and original approval remain pending.',
                'Short synthetic schedules preserve all configured updates; no historical duration, accuracy or TensorFlow seeded replay claim.',
                'Page/calendar policy and three-model lifecycle remain explicit recipe-specific operations. Numeric correlation, normalization and output alignment are separately reusable.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/decomposed_webtraffic_graph_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,nodes=result['nodes'],edges=result['edges'],cases=len(result['cases']),approved=False)))
