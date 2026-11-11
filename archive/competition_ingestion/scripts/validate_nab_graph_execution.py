"""Run all NAB phases through the actual serialized graph runner."""
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
from nab_synthetic import payload
import sciona.atoms.ml.nab_execution as provider
from sciona.nab_graph import build_nab_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_nab_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='nab-private-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-nab','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert values['detection']['models_fitted']==90
    assert len(values['detection']['detections'])==100 and any(values['detection']['detections'])
    assert values['evaluation']['scored_points']==85 and values['evaluation']['normalization_defined']
    assert provider.witness_nab_prepare({})=={'kind':'NAB.Prepared'}
    assert provider.witness_nab_execute({'kind':'NAB.Prepared'})=={'kind':'NAB.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('nab_*.py'))]+['scripts/nab_synthetic.py','scripts/validate_nab_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,models_fitted=90,stream_points=100,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual full generic causal isolation-forest synthetic graph execution and corrected NAB evaluation; no served publication, historical winning detector or accuracy claim.')
    (ROOT/'docs/reviews/competition_nab_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
