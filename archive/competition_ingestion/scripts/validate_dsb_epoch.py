"""Compare source training-pass orchestration and full optimizer/model state."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from scripts.audit_competition_dsb_semantics import checked_source
from scripts.validate_dsb_network import reference
from sciona.dsb_training import shared_training_models,run_training_epoch
from sciona.dsb_schedule import CLASSIFIER_PROFILES,DETECTOR_STAGES,DETECTOR_RATES


class SourceCompatibility(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':return node.func.value
        return node
    def visit_Subscript(self,node):
        self.generic_visit(node)
        if (isinstance(node.value,ast.Attribute) and node.value.attr=='data'
                and isinstance(node.slice,ast.Constant) and node.slice.value==0):
            return ast.copy_location(ast.Call(func=ast.Attribute(value=node.value.value,attr='item',ctx=ast.Load()),args=[],keywords=[]),node)
        return node


class SourceSGD(torch.optim.SGD):
    def zero_grad(self):
        return super().zero_grad(set_to_none=False)


def training_function(source, name):
    tree=ast.parse(source.replace('async = True','non_blocking = True'))
    rate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_lr')
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    loop=next(n for n in train.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
    stop=next(i for i,n in enumerate(loop.body) if ast.unparse(n)=='optimizer.step()')
    loop.body=loop.body[:stop+1]
    train.body=train.body[:train.body.index(loop)+1]
    tree=ast.fix_missing_locations(SourceCompatibility().visit(ast.Module(body=[rate,train],type_ignores=[])))
    scope=dict(torch=torch,nn=torch.nn,np=np,time=time,Variable=lambda value:value,
               binary_cross_entropy=torch.nn.functional.binary_cross_entropy)
    exec(compile(tree,'<pinned-source-training-pass>','exec'),scope)
    return scope[name]


def assert_state(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_state(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_state(x,y)
    else:assert a==b


def validate(root,source_root):
    torch.set_num_threads(1)
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    detector_train=training_function(checked_source(source_root,pins,'training/classifier/trainval_detector.py'),'train_nodulenet')
    classifier_train=training_function(checked_source(source_root,pins,'training/classifier/trainval_classifier.py'),'train_casenet')
    layers=ast.parse(checked_source(source_root,pins,'training/classifier/layers.py'))
    layers.body=[n for n in layers.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Loss','hard_mining']]
    scope=dict(torch=torch,nn=torch.nn)
    exec(compile(ast.fix_missing_locations(SourceCompatibility().visit(layers)),'<source-loss>','exec'),scope)
    main=ast.parse(checked_source(source_root,pins,'training/classifier/main.py'))
    epoch_loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    control=compile(ast.Module(body=epoch_loop.body[:3],type_ignores=[]),'<pinned-source-epoch-order>','exec')
    refs=reference(source_root,pins,'net_classifier.py',{'Net','CaseNet'})
    results=[]
    for epoch,start,variant,freeze in [(30,30,3,False),(31,30,3,False),(121,121,4,True),(160,121,4,True)]:
        torch.manual_seed(17)
        detector,classifier=shared_training_models(2)
        original=refs['CaseNet'](2)
        original.load_state_dict(classifier.state_dict(),strict=True)
        original_detector=original.NoduleNet
        actual_opts=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,weight_decay=.0001) for m in [detector,classifier]]
        source_opts=[SourceSGD(m.parameters(),lr=.01,momentum=.9,weight_decay=.0001) for m in [original_detector,original]]
        images=torch.randn(2,1,16,16,16);coords=torch.randn(2,3,4,4,4)
        labels=torch.zeros(2,4,4,4,3,5);labels[...,0]=-1;labels[:,1,1,1,0,0]=1
        detector_batches=[(images,labels,coords)]
        classifier_batches=[(images[None],coords[None],torch.ones(1,2),torch.ones(1,1))]*6
        profile=CLASSIFIER_PROFILES[variant]
        args=SimpleNamespace(lr=None,debug=False,freeze_batchnorm=freeze,lr_stage=np.array(DETECTOR_STAGES),
             lr_preset=DETECTOR_RATES,lr_stage2=np.array(profile['stages']),lr_preset2=profile['rates'],miss_thresh=.03,miss_ratio=1.)
        controls=dict(epoch=epoch,start_epoch=start,args=args,config2={'startepoch':20},case_net=original,
             nod_net=original_detector,loss=scope['Loss'](2),optimizer=source_opts[0],optimizer2=source_opts[1],
             train_loader_case=classifier_batches,train_loader_nod=detector_batches,val_loader_nod=[],
             val_loader_case=[],all_loader_case=[],train_casenet=classifier_train,train_nodulenet=detector_train,
             validate_nodulenet=lambda *a:None,val_casenet=lambda *a:None)
        torch.manual_seed(91)
        exec(control,controls)
        source_rng=torch.get_rng_state()
        torch.manual_seed(91)
        report=run_training_epoch(detector,classifier,*actual_opts,
            lambda task,flags:detector_batches if task=='detector' else classifier_batches,
            epoch=epoch,start_epoch=start,classifier_variant=variant,freeze_batchnorm=freeze)
        assert_state(classifier.state_dict(),original.state_dict())
        for a,b in zip(actual_opts,source_opts):assert_state(a.state_dict(),b.state_dict())
        assert_state(torch.get_rng_state(),source_rng)
        results.append(dict(epoch=epoch,start_epoch=start,variant=variant,freeze_batchnorm=freeze,
                            exact_model_optimizer_and_rng_state=True,passes=report))
    paths=['sciona/dsb_training.py','sciona/dsb_losses.py','sciona/dsb_schedule.py','sciona/dsb_network.py','scripts/validate_dsb_epoch.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],cases=results,
        limitations=['Source validation/logging/checkpoint operations excluded; comparison covers training passes.',
                     'CUDA moves replaced by CPU identity and obsolete scalar diagnostics ported.',
                     'Source-era zero_grad semantics retained; optimizer kernels use current PyTorch.',
                     'Synthetic batches and weights; no trained-model accuracy claim.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_epoch_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
