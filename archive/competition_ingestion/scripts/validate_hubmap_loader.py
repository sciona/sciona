"""Original dataset getitem and batching parity with synthetic in-memory IO."""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import torch
from sciona.hubmap_loader import prepare_sample,ordered_batches
from scripts.validate_hubmap_augmentation import source_recipe


def encode_synthetic(mask):
    flat=mask.T.flatten();padded=np.concatenate(([0],flat,[0]))
    boundaries=np.flatnonzero(padded[1:]!=padded[:-1])+1
    boundaries[1::2]-=boundaries[::2]
    return ' '.join(str(int(x)) for x in boundaries)


def source_dataset(root,source_root,library_root,images,rles,side,training):
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    extra=json.loads((root/'docs/reviews/competition_hubmap_loader_pins.json').read_text())
    assert extra['commit']==pins['commit']
    for name,digest in extra['files'].items():assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest
    name='src/02_train/dataset.py';assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==pins['files'][name]
    ns=dict(np=np)
    functions=[n for n in ast.parse((source_root/'src/utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='rle2mask']
    assert len(functions)==1
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned-rle>','exec'),ns)
    train,valid=source_recipe(root,source_root,library_root,side)
    ns.update(Dataset=torch.utils.data.Dataset,get_transforms_train=train,get_transforms_valid=valid,
        cv2=SimpleNamespace(imread=lambda key:images[int(key)].copy(),cvtColor=cv2.cvtColor,resize=cv2.resize,
                           COLOR_RGB2BGR=cv2.COLOR_RGB2BGR,INTER_AREA=cv2.INTER_AREA),
        opj=lambda directory,name:name,open=lambda key,mode:nullcontext(rles[int(key)]),pickle=SimpleNamespace(load=lambda value:value))
    classes=[n for n in ast.parse((source_root/name).read_text()).body if isinstance(n,ast.ClassDef)]
    assert len(classes)==1
    exec(compile(ast.Module(body=classes,type_ignores=[]),'<pinned-getitem>','exec'),ns)
    frame=pd.DataFrame(dict(data_path=['']*len(images),filename_img=[str(i) for i in range(len(images))],filename_rle=[str(i) for i in range(len(images))]))
    return ns['HuBMAPDatasetTrain'](frame,dict(resolution=images[0].shape[:2],input_resolution=(side,side)),mode='train' if training else 'valid')


def assert_rng(py,npstate,pr,nr):
    assert py==pr.getstate()
    state=nr.get_state();assert state[0]==npstate[0] and state[2:]==npstate[2:]
    np.testing.assert_array_equal(state[1],npstate[1])


def validate(root,source_root,library_root):
    saved_py=random.getstate();saved_np=np.random.get_state();cases=0;batch_cases=0
    try:
        for height,width,side in [(64,64,32),(96,128,64),(128,128,320)]:
            images=[np.random.RandomState(i+179).randint(0,256,(height,width,3)).astype(np.uint8) for i in range(3)]
            masks=[np.zeros((height,width),dtype=np.uint8),np.ones((height,width),dtype=np.uint8),
                   (np.indices((height,width)).sum(0)%7==0).astype(np.uint8)]
            rles=[encode_synthetic(mask) for mask in masks]
            for training in [False,True]:
                dataset=source_dataset(root,source_root,library_root,images,rles,side,training)
                for seed in range(4):
                    pr=random.Random(seed);nr=np.random.RandomState(seed+5)
                    for i in range(3):
                        random.setstate(pr.getstate());np.random.set_state(nr.get_state())
                        expected=dataset[i];p=random.getstate();n=np.random.get_state()
                        result=prepare_sample(images[i],rles[i],input_side=side,training=training,python_rng=pr,numpy_rng=nr)
                        for value,key in zip(result,['img','mask','label']):torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
                        assert_rng(p,n,pr,nr);cases+=1
                # Replay a source DataLoader with fixed sampler: worker scheduling
                # is excluded, while collation and drop_last are real PyTorch.
                order=[2,0,2,1,0];pr=random.Random(191);nr=np.random.RandomState(193)
                random.setstate(pr.getstate());np.random.set_state(nr.get_state())
                loader=torch.utils.data.DataLoader(dataset,batch_size=2,sampler=order,num_workers=0,drop_last=training,
                    generator=torch.Generator().manual_seed(197))
                expected=list(loader);p=random.getstate();n=np.random.get_state()
                result=ordered_batches(images,rles,order,batch_size=2,input_side=side,training=training,python_rng=pr,numpy_rng=nr)
                assert len(result)==len(expected)
                for actual,wanted in zip(result,expected):
                    for value,key in zip(actual,['img','mask','label']):torch.testing.assert_close(value,wanted[key],rtol=0,atol=0)
                assert_rng(p,n,pr,nr);batch_cases+=1
    finally:
        random.setstate(saved_py);np.random.set_state(saved_np)
    paths=['sciona/hubmap_loader.py','sciona/hubmap_augmentation.py','scripts/validate_hubmap_loader.py',
        'scripts/validate_hubmap_augmentation.py','tests/test_hubmap_loader.py','docs/reviews/competition_hubmap_loader_pins.json',
        'docs/reviews/competition_hubmap_augmentation_pins.json','docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False,synthetic_only=True,exact_getitem_image_mask_label_rng_cases=cases,exact_ordered_batch_cases=batch_cases,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Source file reads/pickle IO replaced with synthetic decoded arrays and RLE text; codec fidelity is not claimed.',
            'Valid ordered in-range RLE only; malformed runs reject explicitly.',
            'Fixed sampler and single-process reference loader; historical shuffle/multiworker scheduling excluded.',
            'Complete balanced-loader-to-epoch and raw tiling/inference graph remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--library-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root,args.library_root)
    (root/'docs/reviews/competition_hubmap_loader.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
