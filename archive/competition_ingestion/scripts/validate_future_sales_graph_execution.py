"""Run all FutureSales phases through the actual serialized graph runner."""
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
from future_sales_synthetic import payload
import sciona.atoms.ml.future_sales_execution as provider
from sciona.future_sales_graph import build_future_sales_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_future_sales_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='future_sales-private-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-future_sales','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert (values['training_rows'],values['validation_rows'],values['refit_rows'],values['forecast_rows'])==(40,16,56,8)
    assert len(values['trials'])==3 and len(values['predictions'])==8 and all(0<=v<=20 for v in values['predictions'])
    assert provider.witness_future_sales_prepare({})=={'kind':'FutureSales.Prepared'}
    assert provider.witness_future_sales_execute({'kind':'FutureSales.Prepared'})=={'kind':'FutureSales.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('future_sales_*.py'))]+['scripts/future_sales_synthetic.py','scripts/validate_future_sales_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,training_rows=40,validation_rows=16,refit_rows=56,forecast_rows=8,trials=3,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual synthetic aggregation, causal features, TPE search, early stopping, final refit and one-period forecast graph; no served publication or historical accuracy claim.')
    (ROOT/'docs/reviews/competition_future_sales_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
