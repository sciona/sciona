"""Run all Cornell phases through the actual serialized graph runner."""
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cornell_synthetic import payload
import sciona.atoms.ml.cornell_execution as provider
from sciona.cornell_graph import build_cornell_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_cornell_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported();torch.set_num_threads(2)
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='cornell-private-graph-') as temporary:
        with patch.dict(os.environ,{'SCIONA_CORNELL_SOURCE_DIR':'/private/tmp/sciona_cornell_source','SCIONA_CORNELL_DEPENDENCY_DIR':'/private/tmp/sciona_cornell_dependencies/audiomentations'}),patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-cornell','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert values['models_completed']==13 and len(values['training_phases'])==17
    assert sum(r['phase']=='continuation' for r in values['training_phases'])==4
    assert len(values['window_votes'])==6 and len(values['whole_record_decisions'])==264
    assert provider.witness_cornell_prepare({})=={'kind':'Cornell.Prepared'}
    assert provider.witness_cornell_execute({'kind':'Cornell.Prepared'})=={'kind':'Cornell.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('cornell_*.py'))]+['scripts/cornell_synthetic.py','scripts/validate_cornell_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,models=13,training_phases=17,continuations=4,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual full synthetic graph execution with two epochs per phase and random offline initialization; no served publication or historical accuracy claim.')
    (ROOT/'docs/reviews/competition_cornell_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
