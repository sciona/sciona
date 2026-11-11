"""Pinned DFDC loss block and scheduler checks using synthetic tensors only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import torch
from torch.optim.lr_scheduler import _LRScheduler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_loss import balanced_bce
from sciona.dfdc_scheduler import PolyLR


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    for name in ('training/pipelines/train_classifier.py','training/tools/schedulers.py','configs/b7.json'):
        assert hashlib.sha256((args.source_root/name).read_bytes()).hexdigest()==next(r['sha256'] for r in pins['files'] if r['path']==name)
    config=json.loads((args.source_root/'configs/b7.json').read_text())
    assert config['losses']=={'BinaryCrossentropy':1} and not config.get('ohem_samples')
    train=next(n for n in ast.parse((args.source_root/'training/pipelines/train_classifier.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_epoch')
    loop=next(n for n in train.body if isinstance(n,ast.For))
    assigned=lambda n,name:isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
    start=next(i for i,n in enumerate(loop.body) if assigned(n,'fake_loss'))
    end=next(i for i,n in enumerate(loop.body) if assigned(n,'loss'))
    block=compile(ast.Module(body=loop.body[start:end+1],type_ignores=[]),'<source-dfdc-loss>','exec')
    comparisons=0
    for dtype in (torch.float16,torch.float32,torch.float64):
        for targets in ([.01,.99,.99,.99],[.01,.01,.01,.99],[.99,.99],[.01,.01],[.5,.99]):
            x=torch.linspace(-1.7,2.1,len(targets),dtype=dtype).reshape(-1,1).requires_grad_()
            y=x.detach().clone().requires_grad_()
            labels=torch.tensor(targets,dtype=torch.float64 if dtype==torch.float64 else torch.float32).reshape(-1,1)
            namespace={'torch':torch,'out_labels':x,'labels':labels,'conf':config,'topk':torch.topk,
                       'loss_functions':{'classifier_loss':torch.nn.BCEWithLogitsLoss()}}
            exec(block,namespace)
            expected=namespace['loss'];actual=balanced_bce(y,labels)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            expected.backward();actual.backward()
            torch.testing.assert_close(y.grad,x.grad,rtol=0,atol=0)
            comparisons+=1
    original=next(n for n in ast.parse((args.source_root/'training/tools/schedulers.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PolyLR')
    current=next(n for n in ast.parse((ROOT/'sciona/dfdc_scheduler.py').read_text()).body if isinstance(n,ast.ClassDef))
    assert ast.dump(original)==ast.dump(current)
    namespace={'_LRScheduler':_LRScheduler}
    exec(compile(ast.Module(body=[original],type_ignores=[]),'<source-dfdc-scheduler>','exec'),namespace)
    scheduler_checks=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for max_iter in (7,100500):
            make_optimizer=lambda:torch.optim.SGD([{'params':[torch.nn.Parameter(torch.ones(1))],'lr':.03},
                                                 {'params':[torch.nn.Parameter(torch.ones(1))],'lr':.01}],lr=.01)
            opt=make_optimizer();other_opt=make_optimizer()
            scheduler=PolyLR(opt,max_iter=max_iter);other=namespace['PolyLR'](other_opt,max_iter=max_iter)
            assert scheduler.last_epoch==other.last_epoch==1
            base=[.03,.01]
            formula=lambda step:[value*(1-((step+1)%max_iter)/max_iter)**.9 for value in base]
            assert [g['lr'] for g in opt.param_groups]==formula(0)
            for epoch in range(2):
                for batch in range(3):
                    old_lr=[g['lr'] for g in opt.param_groups]
                    next_counter=(scheduler.last_epoch+1)%max_iter
                    observation=scheduler.get_lr();source_observation=other.get_lr()
                    expected=[value*(1-next_counter/max_iter)**.9 for value in base]
                    assert observation==source_observation==expected
                    assert [g['lr'] for g in opt.param_groups]==old_lr
                    opt.step();other_opt.step()
                    step=batch+epoch*2500
                    scheduler.step(step);other.step(step)
                    assert scheduler.last_epoch==other.last_epoch==(step+1)%max_iter
                    assert [g['lr'] for g in opt.param_groups]==[g['lr'] for g in other_opt.param_groups]==formula(step)
                    scheduler_checks+=1
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_dfdc_loss.py'],cwd=ROOT,check=True)
    files=['sciona/dfdc_loss.py','sciona/dfdc_scheduler.py','tests/test_dfdc_loss.py',
           'scripts/validate_dfdc_training_math.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-training-math-validation.v1','result':'passed','source_commit':pins['commit'],
            'checks':{'loss_source_value_gradient_cases':comparisons,'independent_loss_tests':12,
                      'scheduler_ast_equal':True,'scheduler_source_and_independent_steps':scheduler_checks},
            'semantics':['Separate per-class BCE means averaged with fixed divisor two, including single-class batches.',
                         'Label exactly .5 belongs to real class; labels enter this loss already smoothed.',
                         'get_lr mutates internal counter; logging observes a different value without changing optimizer parameter-group LR.',
                         'Explicit scheduler.step uses batch + epoch*2500; source modulo counter wrap retained.'],
            'limits':'Synthetic loss/scheduler components only. Source OHEM inactive in selected config. Both schedulers use installed Torch base implementation; no historical binary, Apex, data preparation, full epoch or promotion claim.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_training_math.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
