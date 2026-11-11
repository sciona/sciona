"""Compare raw inference windows with original dataset and Albumentations."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import random
import signal
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import torch
from sciona.hubmap_inference_tiles import InferenceTiles
from scripts.validate_hubmap_augmentation import source_recipe


def source_tiles(root,source,library,image,*,resolution,input_resolution,pad_size):
    # Establish verified original library modules; the inference recipe below
    # executes original ToTensorV2 rather than the training ToTensor recipe.
    source_recipe(root,source,library,input_resolution)
    basic=importlib.import_module('albumentations.core.transforms_interface').BasicTransform
    transforms=importlib.import_module('albumentations.augmentations.transforms')
    composition=importlib.import_module('albumentations.core.composition')
    nodes=[n for n in ast.parse((library/'albumentations/pytorch/transforms.py').read_text()).body
           if isinstance(n,ast.ClassDef) and n.name=='ToTensorV2']
    assert len(nodes)==1
    ns=dict(np=np,torch=torch,BasicTransform=basic)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-ToTensorV2>','exec'),ns)
    ns.update(Normalize=transforms.Normalize,Compose=composition.Compose)
    prefix='src/03_generate_pseudo_labels/03_01_pseudo_label_kaggle_data/'
    pins=json.loads((root/'docs/reviews/competition_hubmap_pseudo_pins.json').read_text())
    for name in ['transforms.py','dataset.py']:
        path=source/(prefix+name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files'][prefix+name]
        tree=ast.parse(path.read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) or
               isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'MEAN','STD'} for t in n.targets)]
        if name=='dataset.py':ns['Dataset']=torch.utils.data.Dataset
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-inference-'+name+'>','exec'),ns)
    h,w=image.shape[:2]
    def read(channels,window):
        (y0,y1),(x0,x1)=window
        return np.moveaxis(image[y0:y1,x0:x1],-1,0)
    raster=SimpleNamespace(count=3,height=h,width=w,read=read)
    ns.update(cv2=cv2,opj=lambda *args:'synthetic',rasterio=SimpleNamespace(open=lambda path:raster),
              Window=SimpleNamespace(from_slices=lambda y,x:(y,x)),
              config=dict(INPUT_PATH='',resolution=resolution,input_resolution=input_resolution,pad_size=pad_size))
    return ns['HuBMAPDataset'](0,pd.DataFrame(dict(id=['synthetic'])))


def validate(root,source,library):
    saved=random.getstate();cases=[]
    try:
        for h,w,res,inp,pad in [(12,12,8,8,2),(11,13,8,16,2),(7,9,8,4,0),
                              (3,3,8,16,2),(512,512,1024,320,256),(513,519,1024,320,256)]:
            image=np.random.RandomState(281).randint(0,256,(h,w,3)).astype(np.uint8)
            reference=source_tiles(root,source,library,image,resolution=res,input_resolution=inp,pad_size=pad)
            rng=random.Random(283)
            actual=InferenceTiles(image,resolution=res,input_resolution=inp,pad_size=pad,python_rng=rng)
            assert len(actual)==len(reference)
            for index in range(len(actual)):
                random.setstate(rng.getstate())
                expected=reference[index];expected_rng=random.getstate()
                result=actual[index]
                assert result['p']==expected['p'] and result['q']==expected['q']
                torch.testing.assert_close(result['img'],expected['img'],rtol=0,atol=0)
                assert rng.getstate()==expected_rng
            cases.append(dict(height=h,width=w,resolution=res,input_resolution=inp,padding=pad,tiles=len(actual),exact=True))
    finally:random.setstate(saved)
    paths=['sciona/hubmap_inference_tiles.py','scripts/validate_hubmap_inference_tiles.py',
           'scripts/validate_hubmap_augmentation.py','docs/reviews/competition_hubmap_pseudo_pins.json',
           'docs/reviews/competition_hubmap_source_pins.json','docs/reviews/competition_hubmap_augmentation_pins.json']
    return dict(approved=False,synthetic_only=True,cases=cases,exact_tensors_coordinates_python_rng=True,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Original dataset with synthetic array-backed raster windows; TIFF/subdataset codecs excluded.',
                             'Original Albumentations0.5.2 under current numerical libraries; historical binaries excluded.',
                             'Inference input stage only; full network shortcut/checkpoint and full CDG gates remain.'])


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--library-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root,args.library_root)
    (root/'docs/reviews/competition_hubmap_inference_tiles.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
