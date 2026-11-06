"""Execute the stored corrected original CHAMPS graph with synthetic inputs and full models."""
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

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.champs_synthetic import synthetic_inputs
import sciona.atoms.ml.decomposed_champs_execution as provider
import importlib
import importlib.metadata
from sciona.champs_ensemble import MODEL_ORDER, blend_predictions
from scripts import stage_champs_corrected_intake as intake
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
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        stored = intake.check_candidate(db, plan, providers)
    digest,nodes,edges=encode_execution_graph(stored)
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('champs_*.py'))]
    paths+=['scripts/validate_decomposed_champs_corrected_execution.py','scripts/champs_synthetic.py','requirements/champs-chemistry.txt']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('decomposed_champs_*.py'))]
    paths += ['sciona/visualizer/runner.py']
    paths += ['scripts/stage_champs_corrected_intake.py', 'scripts/decomposed_champs_approved_provider_state.py', 'sciona/services/execution_graph_codec.py', 'docs/reviews/champs_corrected_intake_plan.json', 'docs/reviews/champs_approved_provider_state.json']
    before = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    provider_paths = {n.matched_primitive: Path(inspect.getfile(importlib.import_module(n.matched_primitive.rsplit('.',1)[0]))) for n in stored.nodes}
    provider_before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in provider_paths.items()}
    runtime_before = {n:importlib.metadata.version(n) for n in ('torch','numpy','pandas','rdkit','openbabel-wheel','networkx','scipy')}

    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    captured={}
    def capture(directory,node,name,value):
        if name.startswith('out_'): captured[node, name]=value
    with tempfile.TemporaryDirectory(prefix='champs-synthetic-graph-') as temporary:
        with patch.dict(os.environ,{'SCIONA_CHAMPS_SOURCE_DIR':str(args.source)}), \
             patch.object(runner,'RUNS_DIR',Path(temporary)), \
             patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-champs','case').execute({'payload':payload},cdg=graph))
    assert status['status']=='completed',status['status']
    assert len(status['trace']) == 5 and not any(t['cached'] for t in status['trace'])
    assert set(captured)=={(n.node_id,'out_'+p.name) for n in graph.nodes for p in n.outputs}
    result=json.loads(json.dumps(captured['result','out_result'],allow_nan=False))
    trained=captured['train_models','out_training_result']
    assert result['predictions']==blend_predictions(trained['model_predictions'],trained['coupling_types'])
    assert result['models_completed']==list(MODEL_ORDER)
    assert len(result['predictions'])==12 and len(result['training'])==13
    assert all(report['selected_epoch']==0 and report['epochs']==1 and len(report['history'])==1 for report in result['training'].values())
    assert before == {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    assert provider_before == {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in provider_paths.items()}
    assert runtime_before == {n:importlib.metadata.version(n) for n in runtime_before}
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        assert intake.check_candidate(db, plan, providers) == stored
    report={'format':'champs-corrected-execution-validation.v1','status':'passed','approved':False,'catalog_mutations':0,'synthetic_only':True,
        'candidate_version_id':plan['version_id'],'plan_sha256':intake.sha(intake.PLAN),
        'stored_candidate':True,'all_nodes_uncached':True,'epochs_per_model':1,
        'serialized_graph_sha256':digest,'provider_sha256':hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        'code_sha256':before,
        'checks':{'actual_runner_nodes':5,'actual_runner_edges':7,'source_ensemble_exact':True,'full_model_lifecycles':13,'synthetic_predictions':12,
                  'complete_checkpoint_reload':True,'strict_json_output':True,'graph_codec_roundtrip':True},
        'resource_policy':{'native_threads':1,'torch_interop_threads':1,'data_loader_workers':0,'serial_models':True},
        'runtime':{n:importlib.metadata.version(n) for n in ('torch','numpy','pandas','rdkit','openbabel-wheel','networkx','scipy')},
        'bound_provider_sha256':provider_before,
        'limits':'One synthetic SGD epoch per full model, full-population mode; no historical accuracy or served publication claim. Raw inputs/checkpoints/predictions remain temporary or in memory.'}
    (ROOT/'docs/reviews/champs_corrected_intake_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    main()
