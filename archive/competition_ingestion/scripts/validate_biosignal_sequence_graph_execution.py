"""Run all BiosignalSequence phases through the actual serialized graph runner."""
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
from biosignal_sequence_synthetic import payload
import sciona.atoms.ml.biosignal_sequence_execution as provider
from sciona.biosignal_sequence_graph import build_biosignal_sequence_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_biosignal_sequence_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='biosignal_sequence-private-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-biosignal_sequence','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert (values['training_rows'],values['calibration_rows'],values['query_rows'])==(4,4,2)
    assert values['epochs']==40 and values['training_windows']==18 and values['query_windows']==7
    scores=np.asarray(values['scores'])
    assert scores.shape==(2,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    assert values['classes']==(scores>=values['threshold']).astype(int).tolist()
    assert values['final_training_loss']<values['initial_training_loss']*.25
    assert provider.witness_biosignal_sequence_prepare({})=={'kind':'BiosignalSequence.Prepared'}
    assert provider.witness_biosignal_sequence_execute({'kind':'BiosignalSequence.Prepared'})=={'kind':'BiosignalSequence.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('biosignal_sequence_*.py'))]+['scripts/biosignal_sequence_synthetic.py','scripts/validate_biosignal_sequence_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,training_rows=4,calibration_rows=4,query_rows=2,epochs=40,training_windows=18,query_windows=7,training_improved=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual synthetic five-stage windows/PSD/dual-CNN/masking/aggregation-threshold execution; no served publication or historical accuracy claim.')
    (ROOT/'docs/reviews/competition_biosignal_sequence_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
