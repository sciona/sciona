"""Independent pinned training/validation across consecutive synthetic epochs."""
import argparse
import ast
import hashlib
import io
import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts.audit_competition_dsb_semantics import checked_source
from scripts.validate_dsb_epoch import SourceCompatibility,SourceSGD,training_function,assert_state
from scripts.validate_dsb_validation import source_function
from scripts.validate_dsb_network import reference
from sciona.dsb_training import shared_training_models
from sciona.dsb_training_range import run_training_range
from sciona.dsb_schedule import CLASSIFIER_PROFILES,DETECTOR_STAGES,DETECTOR_RATES


def validate(root,source_root):
    torch.set_num_threads(1)
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    sources={name:checked_source(source_root,pins,'training/classifier/'+name)
             for name in ['trainval_detector.py','trainval_classifier.py','layers.py','main.py']}
    scope=dict(np=np,torch=torch,nn=torch.nn,time=time,Variable=lambda value,**kwargs:value,
               binary_cross_entropy=torch.nn.functional.binary_cross_entropy)
    layers=ast.parse(sources['layers.py'])
    layers.body=[n for n in layers.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Loss','hard_mining']]
    exec(compile(ast.fix_missing_locations(SourceCompatibility().visit(layers)),'<source-loss>','exec'),scope)
    det_train=training_function(sources['trainval_detector.py'],'train_nodulenet')
    case_train=training_function(sources['trainval_classifier.py'],'train_casenet')
    det_val=source_function(sources['trainval_detector.py'],'validate_nodulenet',scope)
    case_val=source_function(sources['trainval_classifier.py'],'val_casenet',scope)
    main=ast.parse(sources['main.py'])
    loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    control=compile(ast.Module(body=loop.body[:3],type_ignores=[]),'<source-lifecycle-control>','exec')
    refs=reference(source_root,pins,'net_classifier.py',{'Net','CaseNet'})
    reports=[]
    for start,end,freeze in [(49,51,False),(159,161,True)]:
        torch.manual_seed(73)
        detector,classifier=shared_training_models(2)
        original=refs['CaseNet'](2);original.load_state_dict(classifier.state_dict())
        actual_opts=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,weight_decay=.0001) for m in [detector,classifier]]
        source_opts=[SourceSGD(m.parameters(),lr=.01,momentum=.9,weight_decay=.0001) for m in [original.NoduleNet,original]]
        def detector_batch(size):
            images=torch.randn(size,1,16,16,16);coords=torch.randn(size,3,4,4,4)
            labels=torch.zeros(size,4,4,4,3,5);labels[...,0]=-1;labels[:,1,1,1,0,0]=1
            return images,labels,coords
        def classifier_batch(size):
            return (torch.randn(size,2,1,16,16,16),torch.randn(size,2,3,4,4,4),
                    torch.ones(size,2),(torch.arange(size)%2).float()[:,None])
        train_det=[detector_batch(2)];train_case=[classifier_batch(1)]*6
        val_det=[detector_batch(1),detector_batch(3)]
        val_case=[classifier_batch(1),classifier_batch(3)]
        val_training=[classifier_batch(3),classifier_batch(1)]
        args=SimpleNamespace(lr=None,debug=False,freeze_batchnorm=freeze,lr_stage=np.array(DETECTOR_STAGES),
            lr_preset=DETECTOR_RATES,lr_stage2=np.array(CLASSIFIER_PROFILES[4]['stages']),
            lr_preset2=CLASSIFIER_PROFILES[4]['rates'],miss_thresh=.03,miss_ratio=1.)
        observed=[]
        def validate_detector(batches,model,loss):
            with torch.inference_mode():value=det_val(batches,model,loss)
            observed.append(('detector',value['metrics']))
        def validate_classifier(epoch,model,batches,args):
            with torch.inference_mode():value=case_val(epoch,model,batches,args)
            observed.append(('classifier_validation' if batches is val_case else 'classifier_training_validation',value))
        controls=dict(start_epoch=start,args=args,config2={'startepoch':20},case_net=original,
            nod_net=original.NoduleNet,loss=scope['Loss'](2),optimizer=source_opts[0],optimizer2=source_opts[1],
            train_loader_case=train_case,train_loader_nod=train_det,val_loader_nod=val_det,
            val_loader_case=val_case,all_loader_case=val_training,train_casenet=case_train,
            train_nodulenet=det_train,validate_nodulenet=validate_detector,val_casenet=validate_classifier)
        torch.manual_seed(79)
        for epoch in range(start,end+1):controls['epoch']=epoch;exec(control,controls)
        expected_rng=torch.get_rng_state()
        torch.manual_seed(79)
        result=run_training_range(dict(detector=detector,classifier=classifier,detector_optimizer=actual_opts[0],
            classifier_optimizer=actual_opts[1],start_epoch=start),lambda task,profile:train_det if task=='detector' else train_case,
            lambda task:{'detector':val_det,'classifier_validation':val_case,'classifier_training_validation':val_training}[task],
            end_epoch=end,classifier_variant=4,freeze_batchnorm=freeze)
        assert_state(classifier.state_dict(),original.state_dict())
        for actual,expected in zip(actual_opts,source_opts):assert_state(actual.state_dict(),expected.state_dict())
        assert_state(torch.get_rng_state(),expected_rng)
        validations=[v for epoch in result['epochs'] for v in epoch['validation']]
        assert len(validations)==len(observed)
        for actual,(task,expected) in zip(validations,observed):
            assert actual['task']==task;metrics=actual['metrics']
            if task=='detector':
                np.testing.assert_array_equal(metrics['mean_losses'],expected[:,:6].mean(axis=0))
                for key,index in [('positive_correct',6),('positive_count',7),('negative_correct',8),('negative_count',9)]:
                    assert metrics[key]==expected[:,index].sum()
            else:
                for key,name in [('classification_loss','mean_loss2'),('miss_loss','mean_missloss'),('accuracy','mean_acc'),
                    ('true_positives','tpn'),('false_positives','fpn'),('false_negatives','fnn')]:assert metrics[key]==expected[name]
        checkpoint=torch.load(io.BytesIO(result['checkpoint']),weights_only=True)
        assert checkpoint['epoch']==end;assert_state(checkpoint['state_dict'],original.state_dict())
        reports.append(dict(start_epoch=start,end_epoch=end,freeze_batchnorm=freeze,
            exact_model_optimizer_rng_checkpoint_and_validation=True,validation_passes=len(observed)))
    paths=['sciona/dsb_training.py','sciona/dsb_training_range.py','sciona/dsb_network.py',
           'sciona/dsb_losses.py','sciona/dsb_validation.py','sciona/dsb_schedule.py','sciona/dsb_checkpoint.py',
           'scripts/validate_dsb_epoch.py','scripts/validate_dsb_validation.py','scripts/validate_dsb_network.py',
           'scripts/validate_dsb_source_lifecycle.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],cases=reports,
        limitations=['Synthetic batches supplied directly; raw input preparation and serialized runner are not compared here.',
            'Historical CPU/CUDA, Variable and zero_grad adaptations match earlier source validators.',
            'Source logging and checkpoint file IO omitted; final checkpoint tensors compared directly.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_source_lifecycle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
