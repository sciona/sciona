"""Full network, original loss statements and persistent Adam state parity."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import torch
from sciona.hubmap_network import UNET_SERESNEXT101
from sciona.hubmap_training import train_batch
from scripts.validate_hubmap_network import source_model
from scripts.validate_hubmap_losses import reference as source_loss


def exact(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def validate(root,source_root):
    torch.set_num_threads(1)
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    original=source_model(root,source_root);criterion=source_loss(source_root,pins)
    torch.manual_seed(127);candidate=UNET_SERESNEXT101((32,32),True,True,None,load_weights=False)
    torch.manual_seed(127);reference=original((32,32),True,True,None,load_weights=False)
    optimizers=[torch.optim.Adam(m.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-5) for m in [candidate,reference]]
    reports=[]
    for step in range(2):
        torch.manual_seed(131+step)
        images=torch.randn(2,3,32,32)
        masks=(torch.rand(2,1,32,32)>.8).float()
        if step==0:masks[0].zero_()
        else:masks.zero_()
        labels=torch.tensor([0.,1.]) if step==0 else torch.zeros(2)
        before=torch.get_rng_state()
        actual=train_batch(candidate,optimizers[0],images,masks,labels)
        actual_rng=torch.get_rng_state()
        torch.set_rng_state(before)
        reference.train();optimizers[1].zero_grad()
        logits,deep,classification=reference(images)
        expected=criterion(logits,masks,deep,classification,labels,dict(deepsupervision=True,clfhead=True))
        expected.backward();optimizers[1].step()
        exact(actual,expected.detach());exact(actual_rng,torch.get_rng_state())
        exact(candidate.state_dict(),reference.state_dict())
        exact(optimizers[0].state_dict(),optimizers[1].state_dict())
        reports.append(dict(step=step+1,all_empty_masks=bool(step),exact_loss_model_buffers_optimizer_rng=True))
        del logits,deep,classification,expected
    paths=['sciona/hubmap_training.py','sciona/hubmap_losses.py','sciona/hubmap_encoder.py','sciona/hubmap_network.py',
        'scripts/validate_hubmap_training.py','scripts/validate_hubmap_losses.py','scripts/validate_hubmap_network.py',
        'docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False,synthetic_only=True,cases=reports,parameters=sum(p.numel() for p in candidate.parameters()),
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Current CPU float32 network/loss/Adam parity; CUDA autocast and GradScaler are excluded.',
            'Preprocessed synthetic minibatches only; epoch sampling/augmentation/validation/checkpoint selection and complete graph remain.',
            'Two consecutive optimizer updates; no trained predictive-quality or convergence claim.'])


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
