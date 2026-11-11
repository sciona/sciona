"""Run all VideoMultilabel phases through the actual serialized graph runner."""
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
from video_multilabel_synthetic import payload
import sciona.atoms.ml.video_multilabel_execution as provider
from sciona.video_multilabel_graph import build_video_multilabel_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_video_multilabel_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='video_multilabel-private-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-video_multilabel','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert (values['training_rows'],values['calibration_rows'],values['query_rows'])==(8,4,2)
    assert values['epochs']==80 and values['label_count']==values['sparse_models']==2
    scores=np.asarray(values['scores'])
    assert scores.shape==(2,2) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    assert values['labels']==(scores>=values['thresholds']).astype(int).tolist()
    assert values['final_training_loss']<values['initial_training_loss']*.25
    assert provider.witness_video_multilabel_prepare({})=={'kind':'VideoMultilabel.Prepared'}
    assert provider.witness_video_multilabel_execute({'kind':'VideoMultilabel.Prepared'})=={'kind':'VideoMultilabel.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('video_multilabel_*.py'))]+['scripts/video_multilabel_synthetic.py','scripts/validate_video_multilabel_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,training_rows=8,calibration_rows=4,query_rows=2,epochs=80,label_count=2,sparse_models=2,training_improved=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual synthetic five-stage frame/attention/context/sparse-fusion/threshold execution; no served publication or historical accuracy claim.')
    (ROOT/'docs/reviews/competition_video_multilabel_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
