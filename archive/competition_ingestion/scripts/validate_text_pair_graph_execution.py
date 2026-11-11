"""Run all TextPair phases through the actual serialized graph runner."""
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
from text_pair_synthetic import payload
import sciona.atoms.ml.text_pair_execution as provider
from sciona.text_pair_graph import build_text_pair_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_text_pair_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='text_pair-private-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-text_pair','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert (values['training_rows'],values['calibration_rows'],values['query_rows'])==(8,4,2)
    assert values['models']==2 and values['forest_trees']==64 and values['feature_count']==12
    assert len(values['scores'])==2 and all(0<=v<=1 for v in values['scores'])
    assert values['matches']==[int(v>=values['threshold']) for v in values['scores']]
    assert provider.witness_text_pair_prepare({})=={'kind':'TextPair.Prepared'}
    assert provider.witness_text_pair_execute({'kind':'TextPair.Prepared'})=={'kind':'TextPair.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('text_pair_*.py'))]+['scripts/text_pair_synthetic.py','scripts/validate_text_pair_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,training_rows=8,calibration_rows=4,query_rows=2,models=2,forest_trees=64,feature_count=12,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual synthetic six-stage feature/ensemble/threshold/prediction execution; no served publication or historical accuracy claim.')
    (ROOT/'docs/reviews/competition_text_pair_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
