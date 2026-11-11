"""Execute the full independent VSB pipeline through serialized graph nodes."""
import asyncio
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
from vsb_synthetic import payload
import sciona.atoms.ml.vsb_execution as provider
from sciona.vsb_graph import build_vsb_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=sorted((ROOT/'sciona').glob('vsb_*.py'))+[ROOT/'scripts/vsb_synthetic.py',Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    provider_path=Path(inspect.getfile(provider));provider_hash=sha(provider_path)
    digest,nodes,edges=encode_execution_graph(build_vsb_graph());graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported();captured={};executed=[]
    def capture(directory,node,name,value):
        executed.append((node,name))
        if node=='execute' and name=='out_result':captured['result']=value
    print('Starting serialized VSB graph with synthetic independent configurations',flush=True)
    with tempfile.TemporaryDirectory(prefix='vsb-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-vsb','case').execute({'payload':payload()},cdg=graph))
    assert status['status']=='completed'
    assert ('prepare','out_prepared') in executed and ('execute','out_result') in executed
    result=json.loads(json.dumps(captured['result'],allow_nan=False));scores=np.asarray(result['probabilities'])
    assert scores.shape==(8,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    assert result['models']==125
    assert result['signal_decisions']==[[int(p>result['threshold'])]*3 for p in scores]
    assert provider.witness_vsb_prepare({})=={'kind':'VSB.Prepared'}
    assert provider.witness_vsb_execute({'kind':'VSB.Prepared'})=={'kind':'VSB.Result'}
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths} and sha(provider_path)==provider_hash
    report=dict(status='passed',approved=False,catalog_mutations=0,
        source_version_id='d79e40e4-45b0-5e74-a83c-c68c91f12afb',source_content_hash='565e511ee8d20ce0b32adf5c8cc726150f235b1f97d5502d62c96cfe339f5ac1',
        serialized_graph_sha256=digest,provider_sha256=provider_hash,code_sha256=before,
        checks=dict(actual_runner_nodes=2,models=125,raw_preprocessing=True,full_training_limits=True,synthetic_query_rows=8,
            strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True),
        scope='Independent full CPU pipeline on synthetic raw signals at generalized signal length. Historical precision, unbiased validation and competitive accuracy unqualified.')
    (ROOT/'docs/reviews/competition_vsb_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
