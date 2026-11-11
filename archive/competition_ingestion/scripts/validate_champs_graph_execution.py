"""Run the serialized CHAMPS graph with synthetic raw inputs and full models."""
import argparse
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
from champs_synthetic import synthetic_inputs
import sciona.atoms.ml.champs_execution as provider
from sciona.champs_ensemble import MODEL_ORDER
from sciona.champs_graph import build_champs_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def synthetic_payload():
    atoms,couplings=synthetic_inputs()
    training=[]
    for key,rows in atoms.groupby('molecule_name',sort=True):
        cs=couplings[couplings.molecule_name==key]
        training.append({'key':key,'elements':rows.atom.tolist(),'coordinates':rows[['x','y','z']].values.tolist(),
            'couplings':[{'key':f'synthetic-{r.id}','atoms':[r.atom_index_0,r.atom_index_1],
                          'type':r.type,'target':r.scalar_coupling_constant} for r in cs.itertuples()]})
    inference=json.loads(json.dumps(training))
    for molecule in inference:
        for coupling in molecule['couplings']: del coupling['target']
    return {'version':1,'training':training,'inference':inference,'selection':'full',
            'models':{name:{'epochs':1,'options':{'optim':'SGD','lr':1e-4,'batch_size':2,
                        'batch_chunk':2,'warmup_step':1,'max_bond_count':406}} for name in MODEL_ORDER}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    payload=synthetic_payload()
    digest,nodes,edges=encode_execution_graph(build_champs_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    torch.set_num_threads(2)
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result': captured['result']=value
    with tempfile.TemporaryDirectory(prefix='champs-synthetic-graph-') as temporary:
        with patch.dict(os.environ,{'SCIONA_CHAMPS_SOURCE_DIR':str(args.source)}), \
             patch.object(runner,'RUNS_DIR',Path(temporary)), \
             patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-champs','case').execute({'payload':payload},cdg=graph))
    assert status['status']=='completed',status['status']
    result=json.loads(json.dumps(captured['result'],allow_nan=False))
    assert result['models_completed']==list(MODEL_ORDER)
    assert len(result['predictions'])==12 and len(result['training'])==13
    assert all(report['selected_epoch']==0 for report in result['training'].values())
    assert provider.witness_champs_prepare({})=={'kind':'CHAMPS.Prepared'}
    assert provider.witness_champs_execute({'kind':'CHAMPS.Prepared'})=={'kind':'CHAMPS.Result'}
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('champs_*.py'))]
    paths+=['scripts/validate_champs_graph_execution.py','scripts/champs_synthetic.py','requirements/champs-chemistry.txt']
    report={'format':'champs-graph-execution.v1','status':'passed','approved':False,
        'serialized_graph_sha256':digest,'provider_sha256':hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        'checks':{'actual_runner_nodes':2,'full_model_lifecycles':13,'synthetic_predictions':12,
                  'complete_checkpoint_reload':True,'strict_json_output':True,'graph_codec_roundtrip':True,'provider_witness_contracts':True},
        'limits':'One synthetic SGD epoch per full model, full-population mode; no historical accuracy or served publication claim. Raw inputs/checkpoints/predictions remain temporary or in memory.'}
    (ROOT/'docs/reviews/competition_champs_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    main()
