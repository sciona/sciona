"""Execute the complete Otto synthetic workload through serialized CDG nodes."""
import argparse
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
from otto_synthetic import payload
import sciona.atoms.ml.otto_execution as provider
from sciona.otto_graph import build_otto_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not __debug__:raise RuntimeError('Assertions required for validation')
    parser=argparse.ArgumentParser()
    parser.add_argument('--r-library',required=True);parser.add_argument('--libfm',required=True)
    cli=parser.parse_args()
    paths=sorted((ROOT/'sciona').glob('otto_*.py'))+sorted((ROOT/'sciona').glob('otto_*.R'))
    paths += [ROOT/'scripts/otto_synthetic.py',Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    provider_path=Path(inspect.getfile(provider));provider_hash=sha(provider_path)
    digest,nodes,edges=encode_execution_graph(build_otto_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    captured={};executed=[]
    def capture(directory,node,name,value):
        executed.append((node,name))
        if node=='execute' and name=='out_result':captured['result']=value
    print('Starting complete native workload through serialized Otto graph',flush=True)
    with tempfile.TemporaryDirectory(prefix='otto-synthetic-graph-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-otto','case').execute(
                {'payload':payload(r_library=cli.r_library,libfm=cli.libfm)},cdg=graph))
    assert result['status']=='completed',result['status']
    assert ('prepare','out_prepared') in executed and ('execute','out_result') in executed
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    probabilities=np.asarray(values['blend']['probabilities'])
    assert probabilities.shape==(9,9) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1),1.,rtol=0,atol=1e-8)
    np.testing.assert_array_equal(values['blend']['classes'],np.arange(9))
    assert sum(v['tuning_model_fits'] for v in values['selection'].values())==8800
    assert sum(v['refit_models'] for v in values['selection'].values())==1100
    assert provider.witness_otto_prepare({})=={'kind':'Otto.Prepared'}
    assert provider.witness_otto_execute({'kind':'Otto.Prepared'})=={'kind':'Otto.Result'}
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert sha(provider_path)==provider_hash
    report=dict(status='passed',approved=False,catalog_mutations=0,
        source_version_id='dc4969f7-e5a4-5644-a9d0-fa10493458ed',
        source_content_hash='13be8fd78b963daf9fd05fb5758e1d36fa98d4ed8cfdde85eb829c94f73af30a',
        serialized_graph_sha256=digest,provider_sha256=provider_hash,code_sha256=before,
        checks=dict(actual_runner_nodes=2,meta_tuning_fits=8800,meta_refit_models=1100,
            synthetic_queries_correct=9,strict_json_output=True,graph_codec_roundtrip=True,
            provider_witness_contracts=True,code_unchanged_during_execution=True),
        scope='Complete synthetic independent CPU realization; historical equivalence, GPU, environment closure and publication unqualified.')
    (ROOT/'docs/reviews/competition_otto_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
