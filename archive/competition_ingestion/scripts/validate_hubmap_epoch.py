"""Connected epoch parity using original source loops and full-size network."""
import argparse
import ast
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import torch

from sciona.hubmap_network import UNET_SERESNEXT101
from sciona.hubmap_epoch import run_epoch
from sciona.hubmap_checkpoint import restore_checkpoint
from sciona.hubmap_schedule import CosineLR
from sciona.hubmap_checkpoint_policy import CheckpointPolicy
from scripts.validate_hubmap_network import source_model
from scripts.validate_hubmap_validation import source_validation
from scripts.validate_hubmap_checkpoint_policy import reference as source_policy
from scripts.validate_hubmap_sampling_schedule import source_functions
from scripts.validate_hubmap_training import exact


def source_training(root,source_root,validation):
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    path=source_root/'src/02_train/run.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['src/02_train/run.py']
    loops=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.For) and any(isinstance(s,ast.With) and 'autocast' in ast.unparse(s.items[0].context_expr) for s in n.body)]
    assert len(loops)==1 and len(loops[0].body)==9
    body=copy.deepcopy(loops[0].body[:7])
    # CPU numerical comparison: disable mixed precision and make scaler an
    # identity. Keep original model, metric, loss, backward and Adam statements.
    body[1].items[0].context_expr=ast.parse('torch.enable_grad()',mode='eval').body
    fn=ast.parse('''def source_train(model,optimizer,batches,population,threshold):
    model.train()
    config=dict(clfhead=True,deepsupervision=True,dice_threshold=threshold,lr_scheduler_name='CosineAnnealingLR')
    device='cpu'
    criterion=torch.nn.BCEWithLogitsLoss()
    criterion_clf=torch.nn.BCEWithLogitsLoss()
    scaler=SimpleNamespace(scale=lambda x:x,step=lambda opt:opt.step(),update=lambda:None)
    running_loss_trn=0
    trn_score_numer=0
    trn_score_denom=0
    for data in batches:
        pass
    return dict(loss=running_loss_trn/population,dice=float(trn_score_numer/trn_score_denom),examples=sum(len(b['img']) for b in batches),population=population)
''').body[0]
    loop=next(n for n in fn.body if isinstance(n,ast.For));loop.body=body
    ns=dict(validation.__globals__);ns['SimpleNamespace']=SimpleNamespace
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<pinned-training-loop>','exec'),ns)
    return ns['source_train']


def validate(root,source_root):
    torch.set_num_threads(1)
    original=source_model(root,source_root)
    validate_source=source_validation(root,source_root)
    train_source=source_training(root,source_root,validate_source)
    decisions_source=source_policy(root,source_root)
    _,schedule_source=source_functions(root,source_root)
    torch.manual_seed(163);candidate=UNET_SERESNEXT101((32,32),True,True,None,load_weights=False)
    torch.manual_seed(163);reference=original((32,32),True,True,None,load_weights=False)
    optimizers=[torch.optim.Adam(m.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-5) for m in [candidate,reference]]
    schedulers=[cls(opt,step_size_min=1e-6,t0=19,tmult=1) for cls,opt in zip([CosineLR,schedule_source],optimizers)]
    # Match source restart scheduling before the first executed epoch.
    for _ in range(17):
        for scheduler in schedulers:scheduler.step()
    policy=CheckpointPolicy();history=[];reports=[]
    for epoch in [18,19]:
        torch.manual_seed(167+epoch)
        def batch(n):
            images=torch.randn(n,3,32,32);masks=(torch.rand(n,1,32,32)>.8).float();labels=torch.ones(n)
            if n>1:masks[0].zero_();labels[0]=0
            return images,masks,labels
        training=[batch(2),batch(2)];validation=[batch(2),batch(1)]
        actual=run_epoch(candidate,optimizers[0],schedulers[0],policy,training,validation,epoch=epoch,
            training_population_size=5,training_batch_size=2,validation_population_size=3)
        expected_training=train_source(reference,optimizers[1],[dict(img=i,mask=m,label=l) for i,m,l in training],5,.5)
        expected_validation=validate_source(reference,[dict(img=i,mask=m,label=l) for i,m,l in validation],3,.5)
        history.append((epoch,expected_validation['loss'],expected_validation['dice']))
        config=dict(early_stopping=True,patience=10,lr_scheduler_name='CosineAnnealingLR',lr_scheduler={'CosineAnnealingLR':dict(t0=19)})
        decisions,state=decisions_source(history,config)
        if decisions[-1]['step_scheduler']:schedulers[1].step()
        assert actual['training']==expected_training
        assert actual['validation']==expected_validation
        assert actual['decision']==decisions[-1] and asdict(policy)==state
        exact(candidate.state_dict(),reference.state_dict())
        exact(optimizers[0].state_dict(),optimizers[1].state_dict())
        exact(schedulers[0].state_dict(),schedulers[1].state_dict())
        exact(torch.load(io.BytesIO(actual['checkpoint']),weights_only=True),reference.state_dict())
        # Original torch.save(state_dict) is accepted by the strict runtime loader.
        stream=io.BytesIO();torch.save(reference.state_dict(),stream)
        restore_checkpoint(candidate,stream.getvalue())
        exact(candidate.state_dict(),reference.state_dict())
        reports.append(dict(epoch=epoch,trained_examples=4,sampled_population=5,validation_batch_sizes=[2,1],
            checkpoint_roles=actual['decision']['save'],exact_metrics_model_optimizer_scheduler_checkpoint=True))
        del stream,actual
    paths=sorted(root.glob('sciona/hubmap_*.py'))+sorted(root.glob('scripts/validate_hubmap_*.py'))+[
        root/'docs/reviews/competition_hubmap_source_pins.json',root/'tests/test_hubmap_checkpoint.py']
    return dict(approved=False,synthetic_only=True,cases=reports,
        implementation_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=['Initialized full network and current CPU float32 execution; original CUDA autocast/GradScaler disabled explicitly.',
            'Explicit batches and restart schedule; preparation, sampling-to-loader, augmentation and inference not yet connected.',
            'Model-only warm-start checkpoints; serialized optimizer/RNG resume is not claimed.',
            'No trained quality or whole-CDG approval.'])


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_epoch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(approved=False,cases=report['cases'])))
