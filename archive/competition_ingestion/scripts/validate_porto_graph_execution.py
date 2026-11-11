"""Execute all six independent Porto models through serialized graph nodes."""
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
from porto_synthetic import payload
import sciona.atoms.ml.porto_execution as provider
from sciona.porto_graph import build_porto_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=sorted((ROOT/'sciona').glob('porto_*.py'))+[ROOT/'scripts/porto_synthetic.py',Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    provider_path=Path(inspect.getfile(provider));provider_hash=sha(provider_path)
    digest,nodes,edges=encode_execution_graph(build_porto_graph());graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported();captured={};executed=[]
    def capture(directory,node,name,value):
        executed.append((node,name))
        if node=='execute' and name=='out_result':captured['result']=value
    print('Starting serialized Porto graph with synthetic independent configurations',flush=True)
    with tempfile.TemporaryDirectory(prefix='porto-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-porto','case').execute({'payload':payload()},cdg=graph))
    assert status['status']=='completed'
    assert ('prepare','out_prepared') in executed and ('execute','out_result') in executed
    result=json.loads(json.dumps(captured['result'],allow_nan=False));scores=np.asarray(result['probabilities'])
    assert scores.shape==(8,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    assert result['models']==6 and result['dae_models']==5
    assert result['learned_widths']==[30,8,25,7,45]
    assert result['score_kind']=='equal_probability_mean'
    assert np.mean((scores>=.5)==(np.arange(8)>=4))>=.875
    assert provider.witness_porto_prepare({})=={'kind':'Porto.Prepared'}
    assert provider.witness_porto_execute({'kind':'Porto.Prepared'})=={'kind':'Porto.Result'}
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths} and sha(provider_path)==provider_hash
    report=dict(status='passed',approved=False,catalog_mutations=0,
        source_version_id='ad3239bf-5a38-565a-81da-3d3cba9a02b0',source_content_hash='8b42f3175fa6408c91c325e24784554cc480637a33673bd7a2128e2e657b8dd8',
        serialized_graph_sha256=digest,provider_sha256=provider_hash,code_sha256=before,
        checks=dict(actual_runner_nodes=2,models=6,dae_models=5,synthetic_query_rows=8,
            strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True),
        scope='Independent CPU ensemble with small synthetic configurations. Historical five-model settings, optimizer, CV and competitive accuracy unqualified.')
    (ROOT/'docs/reviews/competition_porto_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
