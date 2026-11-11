"""Train all full source-configured CHAMPS models using the reusable controller."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sciona.champs_ensemble import MODEL_ORDER,blend_predictions
from sciona.champs_prediction import predict_batches
from sciona.champs_source_runtime import ChampsSourceRuntime
from sciona.champs_training import ChampsTrainer,TrainingOptions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    runtime=ChampsSourceRuntime(args.source)
    with tempfile.TemporaryDirectory(prefix='champs-full-training-') as temporary:
        tmp=Path(temporary)
        subprocess.run([sys.executable,str(repo/'scripts/validate_champs_preprocessing_runtime.py'),
            '--source',str(args.source),'--output',str(tmp/'preprocessing.json'),'--tensors',str(tmp/'synthetic.npz')],check=True)
        with np.load(tmp/'synthetic.npz',allow_pickle=False) as archive:
            batch=tuple(torch.from_numpy(archive[f'tensor_{i}'].copy()) for i in range(10))
    torch.set_num_threads(2)
    results=[]
    predictions={}
    inference=list(batch)
    inference[-1]=batch[-1].clone()
    requested=inference[-1][:,:,3]>0
    count=int(requested.sum())
    inference[-1][:,:,0][requested]=torch.arange(count,dtype=torch.float32)
    inference=tuple(inference)
    key_mapping={i:f'synthetic-{i}' for i in range(count)}
    coupling_types={key:'1JHC' for key in key_mapping.values()}

    for variant in MODEL_ORDER:
        model=runtime.create_model(variant)
        options=TrainingOptions(optim='SGD',lr=1e-4,batch_size=2,batch_chunk=2,
            max_step=2,warmup_step=1,max_bond_count=406)
        trainer=ChampsTrainer(runtime,variant,model,options)
        def parameter_digest():
            digest=hashlib.sha256()
            for parameter in model.parameters():
                digest.update(parameter.detach().numpy().tobytes())
            return digest.hexdigest()
        initial=parameter_digest()
        training=trainer.epoch([batch])
        assert torch.isfinite(training[0]), 'nonfinite aggregate training MAE'
        assert initial!=parameter_digest(), 'no model parameter updated'
        validation=trainer.epoch([batch],training=False)
        assert torch.isfinite(validation[0])
        # Sparse synthetic data lack seven target types: original per-type
        # macro metrics are undefined there. Aggregate MAE remains meaningful.
        assert torch.isnan(validation[2]).sum()==7
        assert trainer._namespace['train_step']==1
        predictions[variant]=predict_batches(model,[inference],key_mapping)
        assert len(predictions[variant])==count

        item={'variant':variant,'full_source_model':True,'raw_synthetic_inputs':True,
            'chunked_training_and_evaluation':True,'finite_aggregate_mae':True,
            'parameter_update':True,'undefined_absent_type_metrics':7,'source_precision_predictions':count}
        results.append(item)
        print(json.dumps(item),flush=True)
        del trainer,model,initial,training,validation,parameter_digest
        gc.collect()
    blended=blend_predictions(predictions,coupling_types)
    assert len(blended)==12 and all(np.isfinite(value) for value in blended.values())
    # Independent central-five calculation for the active synthetic type.
    from sciona.champs_ensemble import MODEL_MASK
    for key,value in blended.items():
        selected=sorted(predictions[name][key] for i,name in enumerate(MODEL_ORDER) if MODEL_MASK[i][0])
        assert len(selected)==9
        assert value==float(np.mean(selected[2:7]))
    args.output.write_text(json.dumps({'source_commit':runtime.commit,'results':results,'full_ensemble_predictions':len(blended),'central_five_matches':True,
        'scope':'All thirteen full models, reusable raw preprocessing and source epoch controller; one synthetic SGD epoch, source schedule/clipping and corrected chunked evaluation. Source-precision unscaling and complete 13-model trimmed ensemble pass. Population/selection and served graph still pending.',
        'training_runtime_sha256':hashlib.sha256((repo/'sciona/champs_training.py').read_bytes()).hexdigest(),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':
    main()
