"""Pinned validation reductions on synthetic unequal-sized batches."""
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
from scripts.validate_dsb_epoch import SourceCompatibility
from sciona.dsb_training import shared_training_models
from sciona.dsb_validation import validate_detector,validate_classifier


def source_function(source,name,scope):
    tree=ast.parse(source.replace('async = True','non_blocking = True'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.body=[n for n in fn.body if not (isinstance(n,ast.Expr) and
             (isinstance(n.value,ast.Name) and n.value.id=='print' or
              isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='print'))]
    fn.body.append(ast.parse('return locals()').body[0])
    tree=ast.fix_missing_locations(SourceCompatibility().visit(ast.Module(body=[fn],type_ignores=[])))
    exec(compile(tree,'<pinned-validation-no-print>','exec'),scope)
    return scope[name]


def validate(root,source_root):
    torch.set_num_threads(1)
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    scope=dict(np=np,torch=torch,nn=torch.nn,time=time,Variable=lambda value,**kwargs:value,
               binary_cross_entropy=torch.nn.functional.binary_cross_entropy)
    layers=ast.parse(checked_source(source_root,pins,'training/classifier/layers.py'))
    layers.body=[n for n in layers.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Loss','hard_mining']]
    exec(compile(ast.fix_missing_locations(SourceCompatibility().visit(layers)),'<source-loss>','exec'),scope)
    det=source_function(checked_source(source_root,pins,'training/classifier/trainval_detector.py'),'validate_nodulenet',scope)
    case=source_function(checked_source(source_root,pins,'training/classifier/trainval_classifier.py'),'val_casenet',scope)
    cases=0
    for seed in [3,17]:
        for sizes in [(1,3),(3,1)]:
            torch.manual_seed(seed);detector,classifier=shared_training_models(2)
            detector_batches=[];classifier_batches=[]
            for size in sizes:
                images=torch.randn(size,1,16,16,16);coords=torch.randn(size,3,4,4,4)
                labels=torch.zeros(size,4,4,4,3,5);labels[...,0]=-1;labels[:,1,1,1,0,0]=1
                detector_batches.append((images,labels,coords))
                classifier_batches.append((torch.randn(size,2,1,16,16,16),torch.randn(size,2,3,4,4,4),
                                           torch.ones(size,2),(torch.arange(size)%2).float()[:,None]))
            before={k:v.clone() for k,v in classifier.state_dict().items()}
            with torch.inference_mode():
                original=det(detector_batches,detector,scope['Loss'](2))['metrics']
                expected=case(1,classifier,classifier_batches,SimpleNamespace(miss_thresh=.03))
            actual=validate_detector(detector,detector_batches)
            np.testing.assert_array_equal(actual['mean_losses'],[np.mean(original[:,i]) for i in range(6)])
            for key,i in [('positive_correct',6),('positive_count',7),('negative_correct',8),('negative_count',9)]:
                assert actual[key]==np.sum(original[:,i])
            actual=validate_classifier(classifier,classifier_batches)
            for key,source_key in [('classification_loss','mean_loss2'),('miss_loss','mean_missloss'),
                                    ('accuracy','mean_acc'),('true_positives','tpn'),('false_positives','fpn'),('false_negatives','fnn')]:
                assert actual[key]==expected[source_key],(key,actual[key],expected[source_key])
            for key,value in classifier.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
            cases+=1
    paths=['sciona/dsb_validation.py','tests/test_dsb_lifecycle.py','scripts/validate_dsb_validation.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],
        exact_detector_and_classifier_metric_cases=cases,unequal_batch_sizes_verified=True,
        model_buffers_unchanged=True,
        limitations=['Source printing removed; CUDA/volatile adapted to CPU inference mode and scalar diagnostics ported.',
                     'Undefined source classification rates represented as null, tested separately.',
                     'Model/data quality and historic runtime fidelity are not established.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_validation_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
