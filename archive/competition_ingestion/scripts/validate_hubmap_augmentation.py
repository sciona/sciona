"""Exact image/mask and RNG parity against pinned Albumentations recipe."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import random
import signal
import sys
from types import ModuleType,SimpleNamespace
import numpy as np
import torch
from sciona.hubmap_augmentation import augment


def source_recipe(root,source_root,library_root,side):
    pins=json.loads((root/'docs/reviews/competition_hubmap_augmentation_pins.json').read_text())
    for name,digest in pins['files'].items():
        assert hashlib.sha256((library_root/name).read_bytes()).hexdigest()==digest
    # Bypass only package __init__ re-exports of unused imgaug/domain transforms.
    # Core composition, selected transforms and numerical functions are original.
    for name in ['albumentations','albumentations.augmentations','albumentations.core']:
        if name not in sys.modules:
            module=ModuleType(name);module.__path__=[str(library_root.joinpath(*name.split('.')))];module.__version__='0.5.2'
            sys.modules[name]=module
        else:
            assert list(sys.modules[name].__path__)==[str(library_root.joinpath(*name.split('.')))]
    composition=importlib.import_module('albumentations.core.composition')
    transforms=importlib.import_module('albumentations.augmentations.transforms')
    basic=importlib.import_module('albumentations.core.transforms_interface').BasicTransform
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected optional torchvision normalization')
    import warnings
    tensor_ns=dict(torch=torch,np=np,BasicTransform=basic,warnings=warnings,F=SimpleNamespace(normalize=forbidden))
    names={'img_to_tensor','mask_to_tensor','ToTensor'}
    nodes=[n for n in ast.parse((library_root/'albumentations/pytorch/transforms.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==3
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-ToTensor>','exec'),tensor_ns)
    source_pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    path=source_root/'src/02_train/transforms.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source_pins['files']['src/02_train/transforms.py']
    ns=dict(vars(transforms),Compose=composition.Compose,ToTensor=tensor_ns['ToTensor'],np=np,
        config=dict(input_resolution=(side,side)))
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) or
           isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'MEAN','STD'} for t in n.targets)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-augmentation-recipe>','exec'),ns)
    return ns['get_transforms_train'],ns['get_transforms_valid']


def validate(root,source_root,library_root):
    python_state=random.getstate();numpy_state=np.random.get_state();cases=[]
    try:
        for side in [32,64,320]:
            train,valid=source_recipe(root,source_root,library_root,side)
            image=np.random.RandomState(173).randint(0,256,(side,side,3)).astype(np.uint8)
            mask=(np.indices((side,side)).sum(axis=0)%5==0).astype(np.int8)
            for mode in [False,True]:
                reference=train() if mode else valid()
                for seed in range(16):
                    pr=random.Random(seed);nr=np.random.RandomState(seed+3)
                    random.setstate(pr.getstate());np.random.set_state(nr.get_state())
                    expected=reference(image=image.copy(),mask=mask.copy())
                    expected_py=random.getstate();expected_np=np.random.get_state()
                    actual,target,label=augment(image,mask,training=mode,python_rng=pr,numpy_rng=nr)
                    torch.testing.assert_close(actual,expected['image'],rtol=0,atol=0)
                    torch.testing.assert_close(target,expected['mask'],rtol=0,atol=0)
                    assert torch.equal(label,(expected['mask'].sum()>0).float())
                    assert pr.getstate()==expected_py
                    state=nr.get_state();assert state[0]==expected_np[0] and state[2:]==expected_np[2:]
                    np.testing.assert_array_equal(state[1],expected_np[1])
                    cases.append(dict(side=side,training=mode,seed=seed,exact_image_mask_label_rng=True))
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
    paths=['sciona/hubmap_augmentation.py','scripts/validate_hubmap_augmentation.py',
        'docs/reviews/competition_hubmap_augmentation_pins.json','docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False,synthetic_only=True,cases=cases,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Original Albumentations0.5.2 source under current OpenCV/NumPy/Torch; historical binaries not reproduced.',
            'Unused package re-exports skipped; old ToTensor runs original code with optional torchvision normalization forbidden.',
            'Square RGB uint8 images and binary int8 masks at runtime; no real data used.',
            'Dataset decoding/resizing, DataLoader orchestration and complete graph remain.'])


if __name__=='__main__':
    signal.alarm(90)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--library-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root,args.library_root)
    (root/'docs/reviews/competition_hubmap_augmentation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(approved=False,cases=len(report['cases']),exact_image_mask_label_rng=True)))
