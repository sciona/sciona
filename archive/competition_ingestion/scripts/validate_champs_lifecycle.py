"""Validate full-model population fit, complete checkpoint reload and ensemble."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from champs_synthetic import synthetic_inputs
from sciona.champs_ensemble import MODEL_ORDER,blend_predictions
from sciona.champs_source_runtime import ChampsSourceRuntime
from sciona.champs_preprocessing import ChampsPreprocessor
from sciona.champs_prediction import predict_batches
from sciona.champs_population import population_batches
from sciona.champs_training import TrainingOptions
from sciona.champs_lifecycle import fit_variant,load_for_prediction


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runtime=ChampsSourceRuntime(args.source)
    preprocessor=ChampsPreprocessor(runtime)
    atoms,couplings=synthetic_inputs()
    packed,state=preprocessor.prepare(atoms,couplings)
    mapping={int(i):f'synthetic-{i}' for i in couplings['id']}
    types={mapping[int(row.id)]:row.type for row in couplings.itertuples()}
    predictions,results={},[]
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='champs-lifecycle-') as temporary:
        checkpoint=Path(temporary)/'synthetic-state.pt'
        for variant in MODEL_ORDER:
            report=fit_variant(runtime,variant,packed,state,
                TrainingOptions(optim='SGD',lr=1e-4,batch_size=2,batch_chunk=2,warmup_step=1,max_bond_count=406),
                epochs=1,checkpoint_path=checkpoint)
            assert report['selected_epoch']==0
            gc.collect()
            model,restored_state=load_for_prediction(runtime,variant,checkpoint)
            assert restored_state==state
            inference,_=preprocessor.prepare(atoms,couplings.drop(columns=['scalar_coupling_constant']),state=restored_state,labeled=False)
            predictions[variant]=predict_batches(model,population_batches(inference,batch_size=3,shuffle=False,drop_last=False,seed=0),mapping)
            assert len(predictions[variant])==12
            results.append({'variant':variant,'full_population_fit':True,'complete_checkpoint_reloaded':True,
                            'restored_preprocessing_inference':True,'predictions':12})
            print(json.dumps(results[-1]),flush=True)
            del model,restored_state,inference
            gc.collect()
        blended=blend_predictions(predictions,types)
        assert len(blended)==12 and all(np.isfinite(x) for x in blended.values())
    repo=Path(__file__).resolve().parents[1]
    bound=['champs_lifecycle','champs_training','champs_preprocessing','champs_source_runtime','champs_population','champs_prediction','champs_ensemble']
    args.output.write_text(json.dumps({'source_commit':runtime.commit,'results':results,'ensemble_predictions':12,
        'scope':'All thirteen full models: raw synthetic fit, population batching, one SGD epoch in full mode, complete checkpoint reload, restored preprocessing inference and ensemble. Validation-based selection tested separately; served graph pending.',
        'runtime_hashes':{name:hashlib.sha256((repo/'sciona'/f'{name}.py').read_bytes()).hexdigest() for name in bound},
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':
    main()
