"""Execute the full independent Amex pipeline through serialized graph nodes."""
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
from amex_synthetic import payload
import sciona.atoms.ml.amex_execution as provider
from sciona.amex_graph import build_amex_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=sorted((ROOT/'sciona').glob('amex_*.py'))+[ROOT/'scripts/amex_synthetic.py',Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    provider_path=Path(inspect.getfile(provider));provider_hash=sha(provider_path)
    digest,nodes,edges=encode_execution_graph(build_amex_graph());graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported();captured={};executed=[]
    def capture(directory,node,name,value):
        executed.append((node,name))
        if node=='execute' and name=='out_result':captured['result']=value
    print('Starting serialized Amex graph with synthetic independent configurations',flush=True)
    with tempfile.TemporaryDirectory(prefix='amex-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-amex','case').execute({'payload':payload()},cdg=graph))
    assert status['status']=='completed'
    assert ('prepare','out_prepared') in executed and ('execute','out_result') in executed
    result=json.loads(json.dumps(captured['result'],allow_nan=False));scores=np.asarray(result['scores'])
    assert scores.shape==(8,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    assert result['models']==25
    assert result['model_counts']==dict(row=5,downstream_tree=10,neural=10)
    assert result['score_kind']=='literal_weighted_sum_0.9'
    assert provider.witness_amex_prepare({})=={'kind':'Amex.Prepared'}
    assert provider.witness_amex_execute({'kind':'Amex.Prepared'})=={'kind':'Amex.Result'}
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths} and sha(provider_path)==provider_hash
    report=dict(status='passed',approved=False,catalog_mutations=0,
        source_version_id='d505faaa-7277-5559-b32c-f95c9f1f9d63',source_content_hash='f03e6e0c5dfb38e00baa98ea53f9a45a43b9ac4536c343bac2449ca09dbff138',
        serialized_graph_sha256=digest,provider_sha256=provider_hash,code_sha256=before,
        checks=dict(actual_runner_nodes=2,models=25,raw_preprocessing=True,full_training_limits=True,synthetic_query_rows=8,
            strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True),
        scope='Independent full CPU pipeline on synthetic raw sequences at generalized feature widths. Historical precision, unbiased validation and competitive accuracy unqualified.')
    (ROOT/'docs/reviews/competition_amex_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
