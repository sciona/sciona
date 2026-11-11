"""Execute the complete synthetic Santander workload through serialized nodes."""
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
from santander_synthetic import payload
import sciona.atoms.ml.santander_execution as provider
from sciona.santander_graph import build_santander_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=sorted((ROOT/'sciona').glob('santander_*.py'))+[ROOT/'scripts/santander_synthetic.py',Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths};provider_path=Path(inspect.getfile(provider));provider_hash=sha(provider_path)
    digest,nodes,edges=encode_execution_graph(build_santander_graph());graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported();captured={};executed=[]
    def capture(directory,node,name,value):
        executed.append((node,name))
        if node=='execute' and name=='out_result':captured['result']=value
    print('Starting complete serialized Santander graph with full pseudo-label counts',flush=True)
    with tempfile.TemporaryDirectory(prefix='santander-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-santander','case').execute({'payload':payload()},cdg=graph))
    assert status['status']=='completed'
    assert ('prepare','out_prepared') in executed and ('execute','out_result') in executed
    result=json.loads(json.dumps(captured['result'],allow_nan=False));scores=np.asarray(result['scores'])
    assert scores.shape==(8500,) and np.isfinite(scores).all()
    assert result['model_counts']==dict(initial_neural=10,neural=10,tree=10)
    assert result['selection_counts']=={
        'neural':dict(positive=5000,negative=3000,labeled_rows=8060,query_reference_rows=8500),
        'tree':dict(positive=2700,negative=2000,labeled_rows=4760,query_reference_rows=8500)}
    assert result['score_kind']=='rank_blend' and result['reference_policy']=='retain_selected'
    assert provider.witness_santander_prepare({})=={'kind':'Santander.Prepared'}
    assert provider.witness_santander_execute({'kind':'Santander.Prepared'})=={'kind':'Santander.Result'}
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths} and sha(provider_path)==provider_hash
    report=dict(status='passed',approved=False,catalog_mutations=0,
        source_version_id='743c3225-058e-5a7e-ab18-ae5dd1c2b7d4',source_content_hash='162915709dcde12f33a88f1d693b169aa8509dbd7e5b29755f89a728d388ee11',
        serialized_graph_sha256=digest,provider_sha256=provider_hash,code_sha256=before,
        checks=dict(actual_runner_nodes=2,models=30,synthetic_query_rows=8500,full_default_pseudo_counts=True,
            strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True),
        scope='Complete independent CPU lifecycle on four synthetic feature groups. Historical settings, reference membership, unbiased validation and competitive accuracy unqualified.')
    (ROOT/'docs/reviews/competition_santander_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
