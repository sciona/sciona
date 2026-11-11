"""Run all Geotemporal phases through the actual serialized graph runner."""
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
from geotemporal_synthetic import payload
import sciona.atoms.ml.geotemporal_execution as provider
from sciona.geotemporal_graph import build_geotemporal_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_geotemporal_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='geotemporal-private-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-geotemporal','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert (values['training_rows'],values['validation_rows'],values['warmup_rows'],values['query_rows'])==(9,6,3,2)
    assert values['folds']==2 and values['trees']==32 and values['feature_count']==11
    predictions=np.asarray(values['predictions'])
    assert predictions.shape==(2,) and np.isfinite(predictions).all() and (predictions>=0).all()
    assert np.isfinite(values['validation_mse'])
    assert provider.witness_geotemporal_prepare({})=={'kind':'Geotemporal.Prepared'}
    assert provider.witness_geotemporal_execute({'kind':'Geotemporal.Prepared'})=={'kind':'Geotemporal.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('geotemporal_*.py'))]+['scripts/geotemporal_synthetic.py','scripts/validate_geotemporal_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,training_rows=9,validation_rows=6,warmup_rows=3,query_rows=2,folds=2,trees=32,feature_count=11,nonnegative_predictions=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual synthetic five-stage causal-join/features/forward-validation/tree-fusion/smoothing execution; no served publication or historical accuracy claim.')
    (ROOT/'docs/reviews/competition_geotemporal_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
