"""Execute the exact stored March candidate through the production catalog converter."""
from scripts.validate_decomposed_march_baseline import ROOT, sha, runtime
import asyncio
import inspect
import importlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_info
from sciona.march_contract import prepare, execute
from scripts import stage_march_corrected_intake as intake
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts.march_synthetic import payload


def validate():
    versions = runtime()
    plan, provider_plan = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph = intake.check_candidate(db,plan,provider_plan)
    if encode_execution_graph(graph)[0] != plan['graph_sha256']:
        raise ValueError('Stored candidate digest differs')
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest: raise ValueError('Codec roundtrip differs')
    if len(graph.nodes)!=10 or len(graph.edges)!=16: raise ValueError('Complete graph required')
    providers = {}
    for node in graph.nodes:
        module, name = node.matched_primitive.rsplit('.',1)
        fn = getattr(importlib.import_module(module),name)
        if list(inspect.signature(fn).parameters)!=[p.name for p in node.inputs]:
            raise ValueError('Callable input interface differs')
        providers[node.matched_primitive] = Path(inspect.getfile(inspect.unwrap(fn)))
    if len(providers)!=9: raise ValueError('Nine distinct providers required')
    provider_hashes = {k:sha(p) for k,p in providers.items()}
    paths = ['sciona/decomposed_march_graph.py','sciona/decomposed_march_runtime.py',
        'sciona/march_contract.py','sciona/march_features.py','sciona/march_training.py','sciona/march_graph.py',
        'sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py',
        'scripts/march_synthetic.py','scripts/validate_march_corrected_execution.py',
        'scripts/stage_march_corrected_intake.py','scripts/decomposed_march_approved_provider_state.py',
        'scripts/validate_decomposed_march_baseline.py']
    before = {p:sha(ROOT/p) for p in paths}
    baseline = json.loads((ROOT/'docs/reviews/decomposed_march_baseline_execution.json').read_text())
    for p,h in baseline['code_sha256'].items():
        if sha(ROOT/p)!=h: raise ValueError('Qualified reference changed')
    evidence_paths=['docs/reviews/march_corrected_intake_plan.json','docs/reviews/march_approved_provider_state.json']
    evidence_before={p:sha(ROOT/p) for p in evidence_paths}
    runner._ensure_atoms_imported()
    cases=[]
    for seed in (0,3,11):
        inputs=payload();inputs['seed']=seed
        inputs['prediction']=[dict(season=4,order=10,a=a,b=b) for a in range(3) for b in range(3) if a!=b]
        rng=np.random.default_rng(seed)
        for game in inputs['games']:
            a,b=map(int,rng.integers(45,120,size=2))
            if a==b:b+=1
            game['a_box']['points']=a;game['b_box']['points']=b
            for box in ('a_box','b_box'):
                game[box]['field_goal_attempts']=int(rng.integers(45,80))
                game[box]['free_throw_attempts']=int(rng.integers(5,35))
                game[box]['turnovers']=int(rng.integers(5,15))
        inputs['feature_controls']['elo_scale']={0:200.,3:400.,11:800.}[seed]
        captured={}
        def capture(directory,node,name,value):
            if name.startswith('out_'):captured[node,name]=value
        with tempfile.TemporaryDirectory(prefix='march-synthetic-full-graph-') as directory:
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
                status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-march','decomposed').execute({'payload':inputs},cdg=graph))
        if status['status']!='completed' or len(status['trace'])!=10 or any(t['cached'] for t in status['trace']):
            raise ValueError('All ten nodes must execute uncached: '+str(status))
        if set(captured)!={(n.node_id,'out_'+p.name) for n in graph.nodes for p in n.outputs}:
            raise ValueError('Missing actual node output')
        actual=json.loads(json.dumps(captured['result','out_result'],allow_nan=False))
        expected=json.loads(json.dumps(execute(prepare(inputs)),allow_nan=False))
        if actual!=expected or np.asarray(actual['probabilities'],np.float64).tobytes()!=np.asarray(expected['probabilities'],np.float64).tobytes():
            raise ValueError('Complete reference result differs')
        if captured['fit_scores','out_sample_weight'] is not None: raise ValueError('Unexpected calibration weighting')
        if not np.any(np.abs(np.asarray(actual['probabilities'])-.5)>1e-8):
            raise ValueError('Nontrivial orientation predictions required for wiring qualification')
        if (actual['training_rows'],actual['calibration_rows'],actual['predicted_matchups'])!=(4,4,6):
            raise ValueError('Complete configured populations required')
        probabilities={(row['a'],row['b']):p for row,p in zip(inputs['prediction'],actual['probabilities'],strict=True)}
        if len(probabilities)!=6 or any(abs(p+probabilities[b,a]-1)>1e-14 for (a,b),p in probabilities.items()):
            raise ValueError('Every ordered matchup and orientation symmetry required')
        cases.append(dict(seed=seed,elo_scale=inputs['feature_controls']['elo_scale'],nodes_executed=10,
            all_uncached=True,probability_bytes_equal=True,full_result_equal=True,nontrivial_orientation_predictions=True,
            predicted_matchups=6,all_ordered_pairs=True,orientation_symmetry=True))
    pools=threadpool_info()
    if not pools or any(p['num_threads']!=1 for p in pools): raise ValueError('One-thread native pools required')
    if versions!=runtime() or before!={p:sha(ROOT/p) for p in paths} or provider_hashes!={k:sha(p) for k,p in providers.items()}:
        raise ValueError('Software changed during execution')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if intake.check_candidate(db,plan,provider_plan)!=graph: raise ValueError('Stored candidate changed during execution')
    if evidence_before!={p:sha(ROOT/p) for p in evidence_paths}: raise ValueError('Evidence changed during execution')
    return dict(format='march-corrected-stored-execution.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,stored_catalog_graph_executed=True,version_id=plan['version_id'],artifact_id=plan['artifact_id'],evidence_sha256=evidence_before,serialized_graph_sha256=digest,
        nodes=10,edges=16,distinct_providers=9,cases=cases,code_sha256=before,
        provider_sha256=provider_hashes,runtime=versions,
        resource_policy=dict(native_threads=1,serial_cases=True),
        limits=['Exact stored candidate materialized by the production catalog converter and executed through the native runner; original CDG publication remains pending.',
                'Synthetic reference equality and matrix reuse do not prove historical winner fidelity or cross-domain accuracy.',
                'All nine distinct providers are approved and served at Tier 3; this evidence does not itself approve the original CDG.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/march_corrected_intake_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,nodes=report['nodes'],edges=report['edges'],cases=len(report['cases']),approved=False)))
