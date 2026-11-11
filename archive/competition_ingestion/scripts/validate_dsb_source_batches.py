"""Compare complete reopened training batches with source loader execution."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from sciona.dsb_batches import TrainingBatchFactory
from sciona.dsb_schedule import CLASSIFIER_PROFILES
from scripts.dsb_source_batches import SourceTrainingBatches,SourceValidationBatches
from sciona.dsb_validation_batches import ValidationBatchFactory


def validate(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    volume=np.full((1,48,48,48),100,dtype=np.uint8)
    boxes=np.array([[24.,24.,24.,10.],[18.,18.,18.,12.],[30.,30.,30.,14.]])
    arrays=dict(detector_volumes=[volume],boxes_by_image=[boxes],eligible_image_indices=[0],
        classifier_volumes=[volume]*3,proposals_by_case=[np.column_stack(([3.,2.,1.],boxes))]*3,
        known_by_case=[np.array([1,0,1])]*3,case_labels=[1,0,1])
    options=dict(detector_order=[3,2,0,1],classifier_order=[2,0,1],detector_batch_size=3,
        classifier_batch_size=2,topk=2,detector_crop_size=32,classifier_crop_size=16)
    cases=[]
    for variant in [3,4]:
        for seed in [0,7,41]:
            actual_options={**options,'numpy_rng':np.random.RandomState(seed),'label_rng':random.Random(seed)}
            source_options={**options,'numpy_rng':np.random.RandomState(seed),'label_rng':random.Random(seed)}
            actual=TrainingBatchFactory(**arrays,**actual_options)
            source=SourceTrainingBatches(source_root,pins,arrays,source_options)
            batch_count=0
            for task in ['classifier','detector','classifier']:
                profile=({k:CLASSIFIER_PROFILES[variant][k] for k in ['flip','rotate','swap','scale']}
                         if task=='classifier' else dict(flip=True,rotate=False,swap=False,scale=True))
                expected=list(source(task,profile));observed=list(actual(task,profile))
                assert len(expected)==len(observed)
                for a,b in zip(observed,expected):
                    for x,y in zip(a,b):np.testing.assert_array_equal(x,y.numpy())
                    batch_count+=1
            np.testing.assert_array_equal(actual_options['numpy_rng'].rand(10),source_options['numpy_rng'].rand(10))
            assert actual_options['label_rng'].getstate()==source_options['label_rng'].getstate()
            cases.append(dict(variant=variant,seed=seed,exact_batches=batch_count,exact_rng_continuation=True))
    validation_cases=[]
    data=dict(volumes=[volume]*3,proposals=arrays['proposals_by_case'],boxes=[boxes]*3,labels=[1,0,1])
    collections=dict(detector=dict(volumes=[volume],boxes=[boxes]),classifier_validation=data,
                     classifier_training_validation={**data,'labels':[0,1,0]})
    for seed in [0,7,41]:
        opts={k:v for k,v in options.items() if not k.endswith('_order')}
        a={**opts,'numpy_rng':np.random.RandomState(seed),'label_rng':random.Random(seed)}
        b={**opts,'numpy_rng':np.random.RandomState(seed),'label_rng':random.Random(seed)}
        actual=ValidationBatchFactory(collections,**a);source=SourceValidationBatches(source_root,pins,collections,b)
        count=0
        for task in ['detector','classifier_validation','classifier_training_validation']*2:
            observed=list(actual(task));expected=list(source(task))
            assert len(observed)==len(expected)
            for batch,original in zip(observed,expected):
                for value,reference_value in zip(batch,original):
                    np.testing.assert_array_equal(value,reference_value.numpy())
                    assert value.dtype==reference_value.numpy().dtype
                count+=1
        np.testing.assert_array_equal(a['numpy_rng'].rand(10),b['numpy_rng'].rand(10))
        assert a['label_rng'].getstate()==b['label_rng'].getstate()
        validation_cases.append(dict(seed=seed,exact_batches=count,exact_dtypes_and_rng=True))
    paths=['sciona/dsb_batches.py','sciona/dsb_training.py','sciona/dsb_components.py',
           'sciona/dsb_detector_crop.py','sciona/dsb_augmentation.py','sciona/dsb_anchor_mapping.py',
           'sciona/dsb_validation_batches.py','sciona/dsb_intake.py',
           'scripts/dsb_source_batches.py','scripts/validate_dsb_source_batches.py','scripts/validate_dsb_validation_samples.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],cases=cases,validation_cases=validation_cases,
        limitations=['Prepared arrays supplied directly; raw preprocessing and optimizer execution are separate checks.',
            'Source loader integer labels compared by value; training model paths perform their dtype conversions.',
            'Explicit RNG and sample orders replace clock reseeding and worker scheduling; identity eligibility mapping only.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_source_batches.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
