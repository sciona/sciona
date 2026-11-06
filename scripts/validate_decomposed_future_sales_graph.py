"""Run complete decomposed forecasting with the existing rolling atom injected."""
import asyncio
import importlib
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from threadpoolctl import threadpool_info
import sciona.atoms.ml.decomposed_future_sales_execution
from sciona.atoms.ml.tabular.gradient_boosting.atoms import rolling_statistics
from sciona import decomposed_future_sales_features as features
from sciona.future_sales_contract import prepare,execute
from sciona.decomposed_future_sales_graph import build_decomposed_future_sales_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
from scripts.future_sales_synthetic import payload
from scripts.audit_decomposed_future_sales_readiness import ROOT,sha


def validate():
    graph=build_decomposed_future_sales_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    graph=decode_execution_graph(nodes,edges,digest)
    if encode_execution_graph(graph)[0]!=digest or (len(nodes),len(edges))!=(9,30):
        raise ValueError('Complete graph codec roundtrip required')
    provider_hashes={}
    for node in graph.nodes:
        module,symbol=node.matched_primitive.rsplit('.',1)
        function=getattr(importlib.import_module(module),symbol)
        if list(inspect.signature(function).parameters)!=[p.name for p in node.inputs]:
            raise ValueError('Provider parameter order differs')
        provider_hashes[node.matched_primitive]=sha(inspect.getfile(inspect.unwrap(function)))
    runner._ensure_atoms_imported()
    cases=[]
    original_lags=features.lag_features
    for seed,trials in [(92,3),(17,4),(61,5)]:
        p=payload();p['training_controls'].update(seed=seed,trials=trials)
        if seed!=92:
            for i,row in enumerate(p['transactions']):row['quantity']=((i*7+seed)%31)-4
            p['feature_controls'].update(lags=[1,3,12],rolling_windows=[2,4,12])
        captured={};calls=[]
        def tracked_lags(targets,lags,rolling_windows,rolling_function):
            if rolling_function is not rolling_statistics:raise ValueError('Registered rolling identity differs')
            def counted(values,width,aggregations):
                calls.append(width)
                return rolling_function(values,width,aggregations)
            return original_lags(targets,lags,rolling_windows,counted)
        def capture(directory,node,name,value):
            if name.startswith('out_'):captured[node,name]=value
        with tempfile.TemporaryDirectory(prefix='future-sales-generated-graph-') as directory:
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture),patch.object(features,'lag_features',side_effect=tracked_lags):
                status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-future-sales','decomposed').execute({'payload':p},cdg=graph))
        if status['status']!='completed':raise ValueError('Complete decomposed graph failed')
        trace={t['node_id']:t for t in status['trace']}
        if set(trace)!={n.node_id for n in graph.nodes} or any(t['cached'] for t in trace.values()) or trace['rolling'].get('callable_reference') is not True:
            raise ValueError('All nine graph nodes must execute uncached, including callable binding')
        if calls!=[w for w in p['feature_controls']['rolling_windows'] for _ in p['entities']]:
            raise ValueError('All entity/window rolling calls required')
        actual=json.loads(json.dumps(captured['refit','out_result'],allow_nan=False))
        expected=execute(prepare(p))
        if actual!=expected or len(actual['trials'])!=trials:raise ValueError('Full reference computation differs')
        if (actual['training_rows'],actual['validation_rows'],actual['refit_rows'],actual['forecast_rows'])!=(40,16,56,8):
            raise ValueError('Complete configured populations required')
        cases.append(dict(seed=seed,trials=trials,runner_nodes=9,concrete_stages=8,
            all_uncached=True,registered_rolling_identity=True,rolling_calls=len(calls),full_result_equal=True,
            training_rows=40,validation_rows=16,refit_rows=56,forecast_rows=8))
    if any(pool['num_threads']!=1 for pool in threadpool_info()):raise ValueError('Native thread budget exceeded')
    for name,digest_before in provider_hashes.items():
        module,symbol=name.rsplit('.',1)
        if sha(inspect.getfile(inspect.unwrap(getattr(importlib.import_module(module),symbol))))!=digest_before:
            raise ValueError('Provider changed during validation')
    paths=['sciona/decomposed_future_sales_'+suffix+'.py' for suffix in ('contract','features','training','transport','worker','graph')]
    paths+=['scripts/validate_decomposed_future_sales_graph.py','scripts/future_sales_synthetic.py',
            'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']
    return dict(format='decomposed-future-sales-graph.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,serialized_graph_sha256=digest,nodes=9,edges=30,distinct_providers=9,cases=cases,
        provider_sha256=provider_hashes,code_sha256={p:sha(ROOT/p) for p in paths},
        resource_policy=dict(serial_cases=True,serial_workers=True,native_threads=1),
        limits=['Full local registered graph execution and reference parity only; catalog provider approval and stored original execution remain pending.',
                'Eight concrete stages and one callable-binding node; rolling is executed once per requested window/entity pair.',
                'No historical winner, empirical accuracy, clean-deployment or source-data equivalence claim.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/decomposed_future_sales_graph_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,nodes=result['nodes'],edges=result['edges'],cases=len(result['cases']),approved=False)))
