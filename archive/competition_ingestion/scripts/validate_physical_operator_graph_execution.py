"""Run all PhysicalOperator phases through the actual serialized graph runner."""
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from physical_operator_synthetic import payload
import sciona.atoms.ml.physical_operator_execution as provider
from sciona.physical_operator_graph import build_physical_operator_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_physical_operator_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='physical_operator-private-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-physical_operator','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert (values['training_rows'],values['validation_rows'],values['query_rows'])==(8,4,2)
    assert values['epochs']==60 and values['grid_points']==16 and 0<=values['best_epoch']<=60
    predictions=np.asarray(values['predictions'])
    assert predictions.shape==(2,16) and np.isfinite(predictions).all() and (predictions>=0).all()
    np.testing.assert_allclose(predictions.mean(1),np.asarray(payload()['query']['states']).mean(1),rtol=1e-12,atol=1e-12)
    assert values['final_training_mse']<values['initial_training_mse']*.1
    assert provider.witness_physical_operator_prepare({})=={'kind':'PhysicalOperator.Prepared'}
    assert provider.witness_physical_operator_execute({'kind':'PhysicalOperator.Prepared'})=={'kind':'PhysicalOperator.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('physical_operator_*.py'))]+['scripts/physical_operator_synthetic.py','scripts/validate_physical_operator_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,training_rows=8,validation_rows=4,query_rows=2,epochs=60,grid_points=16,nonnegative_mass_conservation=True,training_improved=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual synthetic five-stage state/Fourier-training/checkpoint/projection/smoothing execution; no served publication or historical accuracy claim.')
    (ROOT/'docs/reviews/competition_physical_operator_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
