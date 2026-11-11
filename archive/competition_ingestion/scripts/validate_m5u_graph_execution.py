"""Run both uncertainty providers through the serialized graph at full budgets."""
import asyncio
from datetime import date,timedelta
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sciona.atoms.ml.m5u_execution as provider
from sciona.m5u_graph import build_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def payload():
    n=1338;rng=np.random.default_rng(1297)
    units=4+rng.uniform(0,3,(n,6))+np.sin(np.arange(n)[:,None]/7)
    return dict(version=1,units=units.tolist(),prices=np.full_like(units,2.).tolist(),
                roles=[[0,0,category,category,2*category+product] for category in range(3) for product in range(2)],
                partitions=[[0,13],[1,14],[2,15]],
                calendar=dict(dates=[(date(2000,2,1)+timedelta(days=i)).isoformat() for i in range(n+28)],
                              holidays=[[0.] for _ in range(n+28)],state_codes=[0],events=[[0] for _ in range(n+28)]),
                controls=dict(minimum_day=299,seed=514))


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=sorted(ROOT.glob('sciona/m5u_*.py'))+[ROOT/'sciona/m5u_search_parameters.json',Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    provider_path=Path(inspect.getfile(provider));provider_hash=sha(provider_path)
    digest,nodes,edges=encode_execution_graph(build_graph());graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported();captured={};executed=[]
    def capture(directory,node,name,value):
        executed.append((node,name))
        if node=='execute' and name=='out_result':captured['result']=value
    print('Starting complete synthetic uncertainty graph at full source budgets',flush=True)
    with tempfile.TemporaryDirectory(prefix='m5u-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-m5u','case').execute({'payload':payload()},cdg=graph))
    assert status['status']=='completed'
    assert ('prepare','out_prepared') in executed and ('execute','out_result') in executed
    result=json.loads(json.dumps(captured['result'],allow_nan=False))
    predictions=np.asarray(result['forecast']);reports=result['reports']
    assert predictions.shape==(9,28,39) and np.isfinite(predictions).all()
    assert result['hierarchy_indices']==list(range(39)) and len(reports)==14
    assert sum(r['models'] for r in reports.values())==126
    assert sum(r['query_rows'] for r in reports.values())==2730000
    assert provider.witness_m5u_prepare({})=={'kind':'M5U.Prepared'}
    assert provider.witness_m5u_execute({'kind':'M5U.Prepared'})=={'kind':'M5U.Result'}
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths} and sha(provider_path)==provider_hash
    report=dict(status='passed',approved=False,catalog_mutations=0,
                source_version_id='5d2f953e-86bb-57f8-9a97-e0b6a303fd94',source_content_hash='69deb6e73d3e9e16de3fa53ef108b6a2928c0bed6706bf72ba5c8e558a97d8a5',
                serialized_graph_sha256=digest,provider_sha256=provider_hash,code_sha256=before,
                checks=dict(actual_runner_nodes=2,models=126,quantiles=9,horizon=28,query_rows=2730000,
                            full_source_training_and_inference_budgets=True,strict_json_output=True,
                            graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True),
                scope='Complete corrected independent reconstruction through actual serialized graph on synthetic inputs. Explicit aggregate normalization correction; no historical parity claim. Environment and publication gates remain pending.')
    (ROOT/'docs/reviews/competition_m5_uncertainty_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
