"""Run all OpenVaccine phases through the actual serialized graph runner."""
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import validate_openvaccine_models as setup
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from openvaccine_synthetic import payload
import sciona.atoms.ml.openvaccine_execution as provider
from sciona.openvaccine_graph import build_openvaccine_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def main():
    digest,nodes,edges=encode_execution_graph(build_openvaccine_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported();setup.tf.config.threading.set_intra_op_parallelism_threads(2);setup.tf.config.threading.set_inter_op_parallelism_threads(2)
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='openvaccine-private-graph-') as temporary:
        with patch.dict(os.environ,{'SCIONA_OPENVACCINE_SOURCE_DIR':'/private/tmp/sciona_openvaccine_source','SCIONA_OPENVACCINE_FOLD_BINARY':'/private/tmp/sciona_openvaccine_contrafold/src/contrafold','SCIONA_OPENVACCINE_FOLD_PARAMETERS':'/private/tmp/sciona_openvaccine_eternafold_parameters/EternaFoldParams.v1'}),patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-openvaccine','case').execute({'payload':payload()},cdg=graph))
    assert result['status']=='completed',result['status']
    values=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert values['members']==20 and len(values['rounds'])==1
    assert values['rounds'][0]['members']==values['rounds'][0]['supervised_steps']==values['rounds'][0]['pseudo_label_steps']==20
    assert all(setup.np.asarray(values[k]).shape==(2,14,5) and setup.np.isfinite(values[k]).all() for k in ['predictions','teacher_mean','teacher_std'])
    assert provider.witness_openvaccine_prepare({})=={'kind':'OpenVaccine.Prepared'}
    assert provider.witness_openvaccine_execute({'kind':'OpenVaccine.Prepared'})=={'kind':'OpenVaccine.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('openvaccine_*.py'))]+['scripts/openvaccine_synthetic.py','scripts/validate_openvaccine_graph_execution.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2,models=20,refinement_rounds=1,masked_validation=True,distinct_population_lengths=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True),
        scope='Actual raw synthetic graph execution with folding, shared pretraining, twenty initialized models, refinement and checkpoint ensemble; no served publication, historical recipe or accuracy claim.')
    (ROOT/'docs/reviews/competition_openvaccine_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
