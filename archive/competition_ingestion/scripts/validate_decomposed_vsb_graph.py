"""Execute every registered VSB graph stage with full source training limits."""
import asyncio
import copy
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from threadpoolctl import threadpool_info
import sciona.atoms.ml.decomposed_vsb_execution as providers
from sciona.atoms.scipy.signal import lfilter
from sciona.decomposed_vsb_graph import build_decomposed_vsb_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
from sciona import decomposed_vsb_measurement as measurement
from sciona import decomposed_vsb_training as training
from sciona.vsb_execution import prepare,execute
from scripts.vsb_synthetic import payload
from scripts.audit_decomposed_vsb_readiness import ROOT,sha


def validate():
    paths=[str(p.relative_to(ROOT)) for pattern in ('vsb_*.py','decomposed_vsb_*.py') for p in sorted((ROOT/'sciona').glob(pattern))]
    paths+=['scripts/validate_decomposed_vsb_graph.py','scripts/vsb_synthetic.py','sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']
    before={p:sha(ROOT/p) for p in paths}
    graph=build_decomposed_vsb_graph();digest,nodes,edges=encode_execution_graph(graph)
    graph=decode_execution_graph(nodes,edges,digest)
    if encode_execution_graph(graph)[0]!=digest or len(nodes)!=26:raise ValueError('Complete graph roundtrip required')
    hashes={};functions={}
    for node in graph.nodes:
        fn=lfilter if node.node_id=='filter' else getattr(providers,node.matched_primitive.rsplit('.',1)[-1])
        if list(inspect.signature(fn).parameters)!=[p.name for p in node.inputs]:raise ValueError('Callable inputs differ')
        hashes[node.matched_primitive]=sha(inspect.getfile(inspect.unwrap(fn)));functions[node.matched_primitive]=fn
    runner._ensure_atoms_imported();cases=[]
    expected_ops=(providers.baseline_arguments,lfilter,providers.baseline_residual,providers.window_maxima,
        providers.rank_peaks,providers.amplitude_gradient,providers.select_ranked_peaks,providers.peak_descriptors,providers.phase_quadrants)
    original_collect=measurement.collect;original_ensemble=training.ensemble
    callable_nodes={n.node_id for n in graph.nodes if n.node_id in ('filter','fit_fold') or n.matched_primitive in {fn.__module__+'.'+fn.__name__ for fn in expected_ops}}
    if len(callable_nodes)!=10:raise ValueError('Ten explicit callable nodes required')
    for sign in (1.,-1.):
        raw=payload()
        if sign<0:
            for population in ('training','query'):
                raw[population]['signals']=[[-v for v in row] for row in raw[population]['signals']]
        saved=copy.deepcopy(raw);captured={};columns=[];folds=[]
        def collect(values,coefficients,controls,*ops):
            if tuple(ops)!=expected_ops:raise ValueError('Exact registered numeric callable bindings required')
            columns.append(values.shape[1]);return original_collect(values,coefficients,controls,*ops)
        def ensemble(*args):
            if args[-1] is not providers.fit_fold:raise ValueError('Exact registered fold binding required')
            def fold(*inputs):folds.append(1);return providers.fit_fold(*inputs)
            return original_ensemble(*args[:-1],fold)
        def capture(directory,node,name,value):
            if node=='encode_result' and name=='out_result':captured['result']=value
        with tempfile.TemporaryDirectory(prefix='synthetic-vsb-decomposed-') as directory:
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture),patch.object(measurement,'collect',side_effect=collect),patch.object(training,'ensemble',side_effect=ensemble):
                status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-vsb','decomposed').execute({'payload':raw},cdg=graph))
        if status['status']!='completed':raise ValueError('Complete graph failed: '+str(status.get('error','unknown')))
        traces={t['node_id']:t for t in status['trace']}
        if set(traces)!={n.node_id for n in graph.nodes} or any(t['cached'] for t in traces.values()):raise ValueError('Every graph node must execute uncached')
        if any(traces[n].get('callable_reference') is not True for n in callable_nodes):raise ValueError('Explicit callable references required')
        if sorted(columns)!=[24,600] or len(folds)!=125:raise ValueError('Complete signal and model populations required')
        if captured['result']!=execute(prepare(raw)) or raw!=saved:raise ValueError('Reference output or input preservation differs')
        cases.append(dict(sign=sign,models=len(folds),signal_columns=sum(columns),full_training_limits=True,
            full_result_equal=True,all_nodes_uncached=True,exact_registered_callables=True,root_input_unchanged=True))
    if any(p['num_threads']!=1 for p in threadpool_info()):raise ValueError('Native thread budget exceeded')
    if before!={p:sha(ROOT/p) for p in paths}:raise ValueError('Implementation changed during execution')
    if any(sha(inspect.getfile(inspect.unwrap(functions[n])))!=h for n,h in hashes.items()):raise ValueError('Provider changed')
    return dict(format='decomposed-vsb-graph.v1',passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        graph_sha256=digest,nodes=len(nodes),edges=len(edges),distinct_providers=len(hashes),
        existing_reused_provider='sciona.atoms.scipy.signal.lfilter',new_providers=22,cases=cases,
        provider_sha256=hashes,implementation_sha256=before,resource_policy=dict(serial_cases=True,serial_models=True,native_threads=1),
        limits=['Local registered execution only; new provider catalog qualification and original approval remain pending.',
                'All 125 models and full source stopping limits execute on synthetic populations. No historical accuracy or unbiased validation claim.',
                'Generic numeric and fold operations remain separate from consecutive triples, fixed feature order and repeated-five-way recipe policy.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/decomposed_vsb_graph_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,nodes=result['nodes'],edges=result['edges'],cases=len(result['cases']),approved=False)))
