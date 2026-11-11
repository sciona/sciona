"""Execute the full independent M5 pipeline through serialized graph nodes."""
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
from validate_m5_full_inventory import configuration

def payload():
    config=json.loads(json.dumps(configuration(),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v))
    return dict(version=1,identities=["synthetic-"+str(i) for i in range(70)],configuration=config,controls=dict(recursive_first_day=0,nonrecursive_first_day=710))
import sciona.atoms.ml.m5_execution as provider
from sciona.m5_graph import build_m5_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=sorted((ROOT/'sciona').glob('m5_*.py'))+[ROOT/'scripts/validate_m5_full_inventory.py',Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    provider_path=Path(inspect.getfile(provider));provider_hash=sha(provider_path)
    digest,nodes,edges=encode_execution_graph(build_m5_graph());graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported();captured={};executed=[]
    def capture(directory,node,name,value):
        executed.append((node,name))
        if node=='execute' and name=='out_result':captured['result']=value
    print('Starting serialized M5 graph with synthetic independent configurations',flush=True)
    with tempfile.TemporaryDirectory(prefix='m5-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-m5','case').execute({'payload':payload()},cdg=graph))
    assert status['status']=='completed'
    assert ('prepare','out_prepared') in executed and ('execute','out_result') in executed
    result=json.loads(json.dumps(captured['result'],allow_nan=False));scores=np.asarray(result['forecast'])
    assert scores.shape==(70,28) and np.isfinite(scores).all() and (scores>=0).all()
    assert result['models']==220
    assert result['model_families']==6
    assert result['horizon']==28
    assert provider.witness_m5_prepare({})=={'kind':'M5.Prepared'}
    assert provider.witness_m5_execute({'kind':'M5.Prepared'})=={'kind':'M5.Result'}
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths} and sha(provider_path)==provider_hash
    report=dict(status='passed',approved=False,catalog_mutations=0,
        source_version_id='f64bac1a-ca7f-5f29-a645-ec7b2b42ca4b',source_content_hash='8af8fd4a570aac38f64592903046d4b229fe13604410c786a54ddd9fccc9c949',
        serialized_graph_sha256=digest,provider_sha256=provider_hash,code_sha256=before,
        checks=dict(actual_runner_nodes=2,models=220,raw_preprocessing=True,full_training_limits=True,synthetic_series=70,horizon=28,
            strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True),
        scope='Independent full CPU pipeline on synthetic history with all 220 models and full forecast horizon. Historical sample scale, unbiased validation and competitive accuracy unqualified.')
    (ROOT/'docs/reviews/competition_m5_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
