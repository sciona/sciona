"""Decoded images/RLE through balanced augmented training and retained states."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import random
import signal
import numpy as np
import pandas as pd
import torch
from sciona.hubmap_network import UNET_SERESNEXT101
from sciona.hubmap_schedule import CosineLR
from sciona.hubmap_range import run_training_range
from scripts.validate_hubmap_network import source_model
from scripts.validate_hubmap_validation import source_validation
from scripts.validate_hubmap_epoch import source_training
from scripts.validate_hubmap_checkpoint_policy import reference as source_policy
from scripts.validate_hubmap_sampling_schedule import source_functions
from scripts.validate_hubmap_loader import source_dataset,encode_synthetic,assert_rng
from scripts.validate_hubmap_training import exact


def validate(root,source,library):
    torch.set_num_threads(1)
    original=source_model(root,source);valid_source=source_validation(root,source)
    train_source=source_training(root,source,valid_source);decision_source=source_policy(root,source)
    sample_source,scheduler_source=source_functions(root,source)
    images=[np.random.RandomState(i+199).randint(0,256,(64,64,3)).astype(np.uint8) for i in range(8)]
    present=np.array([False]*3+[True]*5);bins=np.array([0,0,0,1,2,3,4,5])
    rles=[encode_synthetic(np.ones((64,64),dtype=np.uint8) if p else np.zeros((64,64),dtype=np.uint8)) for p in present]
    validation_images=[images[i] for i in [0,3,4]];validation_rles=[rles[i] for i in [0,3,4]]
    torch.manual_seed(211);candidate=UNET_SERESNEXT101((32,32),True,True,None,load_weights=False)
    torch.manual_seed(211);reference=original((32,32),True,True,None,load_weights=False)
    optimizers=[torch.optim.Adam(m.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-5) for m in [candidate,reference]]
    schedulers=[cls(opt,step_size_min=1e-6,t0=19,tmult=1) for cls,opt in zip([CosineLR,scheduler_source],optimizers)]
    pr=random.Random(5);nr=np.random.RandomState(12);tg=torch.Generator().manual_seed(16)
    before_py=pr.getstate();before_np=nr.get_state();before_torch=tg.get_state()
    actual=run_training_range(candidate,optimizers[0],schedulers[0],
        dict(images_bgr=images,rles=rles,present=present,bins=bins),dict(images_bgr=validation_images,rles=validation_rles),
        start_epoch=18,end_epoch=19,maximum_bin=4,training_batch_size=2,validation_batch_size=2,input_side=32,
        python_rng=pr,numpy_rng=nr,torch_generator=tg)
    saved_py=random.getstate();saved_np=np.random.get_state()
    checkpoints={};expected=[];history=[]
    try:
        random.setstate(before_py);np.random.set_state(before_np);refg=torch.Generator().set_state(before_torch)
        frame=pd.DataFrame(dict(is_masked=present,binned=bins,_position=np.arange(8)))
        for _ in range(17):schedulers[1].step()
        for epoch in [18,19]:
            order=sample_source(frame,dict(binned_max=4))
            dataset=source_dataset(root,source,library,[images[i] for i in order],[rles[i] for i in order],32,True)
            batches=list(torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,drop_last=True,num_workers=0,generator=refg))
            training=train_source(reference,optimizers[1],batches,len(order),.5)
            dataset=source_dataset(root,source,library,validation_images,validation_rles,32,False)
            batches=list(torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=0,generator=refg))
            validation=valid_source(reference,batches,3,.5)
            history.append((epoch,validation['loss'],validation['dice']))
            decisions,state=decision_source(history,dict(early_stopping=True,patience=10,lr_scheduler_name='CosineAnnealingLR',
                lr_scheduler={'CosineAnnealingLR':dict(t0=19)}))
            decision=decisions[-1]
            if decision['save']:
                stream=io.BytesIO();torch.save(reference.state_dict(),stream);payload=stream.getvalue()
                for role in decision['save']:checkpoints['snapshot_'+str(epoch) if role=='snapshot' else role]=payload
            if decision['step_scheduler']:schedulers[1].step()
            expected.append(dict(epoch=epoch,training=training,validation=validation,decision=decision))
            if decision['stop']:break
        assert_rng(random.getstate(),np.random.get_state(),pr,nr)
        assert torch.equal(tg.get_state(),refg.get_state())
    finally:
        random.setstate(saved_py);np.random.set_state(saved_np)
    assert actual['epochs']==expected
    exact(candidate.state_dict(),reference.state_dict())
    exact(optimizers[0].state_dict(),optimizers[1].state_dict())
    exact(schedulers[0].state_dict(),schedulers[1].state_dict())
    assert actual['checkpoints'].keys()==checkpoints.keys()
    for key in checkpoints:
        exact(torch.load(io.BytesIO(actual['checkpoints'][key]),weights_only=True),torch.load(io.BytesIO(checkpoints[key]),weights_only=True))
    paths=sorted(root.glob('sciona/hubmap_*.py'))+sorted(root.glob('scripts/validate_hubmap_*.py'))+[
        root/'docs/reviews'/p for p in ['competition_hubmap_source_pins.json','competition_hubmap_loader_pins.json','competition_hubmap_augmentation_pins.json']]
    return dict(approved=False,synthetic_only=True,epochs=[r['epoch'] for r in expected],
        populations=[r['training']['population'] for r in expected],trained_examples=[r['training']['examples'] for r in expected],
        checkpoint_roles=list(checkpoints),exact_metrics_model_optimizer_scheduler_checkpoints_rng=True,
        implementation_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=['Decoded synthetic image/RLE populations and explicit initial weights/RNG; raw tile generation and inference remain.',
            'Current float32 CPU execution and explicit-generator zero-worker DataLoader; historical mixed precision/worker scheduling excluded.',
            'Model-only retained checkpoint parity, not optimizer/RNG serialized resume or learned predictive accuracy.',
            'Full serialized CDG and catalog publication gates remain.'])


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--library-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root,args.library_root)
    (root/'docs/reviews/competition_hubmap_range.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
