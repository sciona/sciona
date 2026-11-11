"""Original training-loop oracle plus full B7 epoch integration on synthetic tensors."""
import argparse
import ast
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_epoch import train_epoch
from sciona.dfdc_scheduler import PolyLR
from sciona.dfdc_classifier import build_classifier


class AverageMeter:
    def __init__(self):self.sum=0.;self.count=0;self.avg=0.
    def update(self,value,n):
        self.sum+=float(value)*n;self.count+=n;self.avg=self.sum/self.count


class Progress:
    def __init__(self,items,observations):self.items=items;self.observations=observations
    def __iter__(self):return iter(self.items)
    def set_postfix(self,values):self.observations.append(dict(values))
    def close(self):pass


class CPUAdaptation(ast.NodeTransformer):
    def __init__(self):self.transfers=0;self.synchronizations=0
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            assert not node.args and not node.keywords
            node.func.attr='cpu';self.transfers+=1
        return node
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='torch.cuda.synchronize':
            self.synchronizations+=1;return None
        return self.generic_visit(node)


def optimizer(model):
    return torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0001,nesterov=True)


def compare_state(a,b):
    assert a.keys()==b.keys()
    for k,v in a.items():
        if isinstance(v,torch.Tensor):torch.testing.assert_close(v,b[k],rtol=0,atol=0)
        elif isinstance(v,dict):compare_state(v,b[k])
        else:assert v==b[k]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    path=args.source_root/'training/pipelines/train_classifier.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==next(r['sha256'] for r in pins['files'] if r['path']=='training/pipelines/train_classifier.py')
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_epoch')
    adaptation=CPUAdaptation();fn=adaptation.visit(fn);ast.fix_missing_locations(fn)
    assert adaptation.transfers==3 and adaptation.synchronizations==1
    events=[];scalars=[]
    namespace={'torch':torch,'AverageMeter':AverageMeter,
               'tqdm':lambda items,**kwargs:Progress(items,events),
               'amp':SimpleNamespace(master_params=lambda opt:(p for g in opt.param_groups for p in g['params']))}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<source-dfdc-epoch-cpu>','exec'),namespace)
    source_epoch=namespace['train_epoch']
    writer=SimpleNamespace(add_scalar=lambda *a,**kw:scalars.append((a,kw)))
    torch.set_num_threads(2)
    cases=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for available,cap in ((1,3),(2,3),(5,3),(4,1)):
            torch.manual_seed(508)
            source=torch.nn.Linear(2,1);model=copy.deepcopy(source)
            aopt,bopt=optimizer(source),optimizer(model)
            asched,bsched=PolyLR(aopt,max_iter=100500),PolyLR(bopt,max_iter=100500)
            batches=[]
            for i in range(available):
                labels=([.01,.99,.99] if i%3==0 else [.99,.99] if i%3==1 else [.01,.01])
                batches.append({'image':torch.randn(len(labels),2)*50,'labels':torch.tensor(labels).reshape(-1,1)})
            conf={'batches_per_epoch':cap,'fp16':False,'optimizer':{'schedule':{'mode':'step'}}}
            for epoch in (0,1):
                events.clear();scalars.clear()
                with redirect_stdout(io.StringIO()):
                    source_epoch(epoch,{'classifier_loss':torch.nn.BCEWithLogitsLoss()},source,aopt,asched,batches,writer,conf,0,False)
                result=train_epoch(model,bopt,bsched,batches,epoch=epoch,batches_per_epoch=cap)
                assert result['optimizer_updates']==len(events)==min(available,cap)
                assert result['examples']==sum(len(b['image']) for b in batches[:cap])
                assert [e['lr'] for e in events]==[e['observed_lr'] for e in result['observations']]
                loss=next(a[1] for a,k in scalars if a[0]=='train/loss')
                assert result['loss']==loss
                compare_state(source.state_dict(),model.state_dict())
                compare_state(aopt.state_dict(),bopt.state_dict())
                compare_state(asched.state_dict(),bsched.state_dict())
                for p,q in zip(source.parameters(),model.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                cases+=1
        with patch('torch.load',side_effect=AssertionError('implicit state load')):
            model=build_classifier(initialization='random',seed=509)
            opt=optimizer(model);scheduler=PolyLR(opt,max_iter=100500)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(510)
                batches=[{'image':torch.rand(1,3,380,380),'labels':torch.tensor([[value]])} for value in (.01,.99)]
                before=model.fc.bias.detach().clone()
                result=train_epoch(model,opt,scheduler,batches,epoch=0)
            assert result['optimizer_updates']==2 and result['examples']==2
            assert not torch.equal(before,model.fc.bias)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert model.encoder.classifier.weight.grad is None
    paths=['sciona/dfdc_epoch.py','sciona/dfdc_loss.py','sciona/dfdc_scheduler.py',
           'sciona/dfdc_classifier.py','sciona/dfdc_stochastic_depth.py','docs/reviews/competition_dfdc_stochastic_depth.json','scripts/validate_dfdc_epoch.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-epoch-validation.v1','result':'passed','source_commit':pins['commit'],
       'checks':{'original_loop_cpu_transfer_edits':3,'original_loop_sync_removals':1,
                 'source_epoch_comparisons':cases,'source_parameters_gradients_momentum_scheduler_exact':True,
                 'full_B7_source_resolution_updates':2,'full_B7_parameters_finite':True},
       'adaptations':['CPU float32 execution; source fp16=False branch and no distributed/Apex synchronization.',
                      'Prepared tensors supplied explicitly; empty epoch and invalid inputs rejected.',
                      'Return aggregate loss/update counts and learning-rate observations rather than source progress logs.'],
       'limits':'Loop parity uses synthetic linear models. Actual B7 executed two synthetic batch1 updates at380resolution, not full default2500batches/40epochs or data-preparation lifecycle. No historical AMP/distributed equivalence, pretrained quality, checkpoint selection or promotion claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_dfdc_epoch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
