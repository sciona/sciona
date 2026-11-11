"""Compare full-size mosaic pixels, boxes, loader order and sampling RNG."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np

from sciona.wheat_mosaic import mosaic_image


def main(source,output):
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!='8224574f23ae68f559e7f9a5a33bbb9e207dfd5ed76f6145db16dfb38c02c84c':
        raise ValueError('historical mosaic source drift')
    tree=ast.parse(raw)
    overlap=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bb_overlap')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WheatDataset')
    nodes=[overlap]+[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'crop_image','load_cutmix_image_and_boxes'}]
    code=compile(ast.Module(body=nodes,type_ignores=[]),'<historical-mosaic>','exec')
    records={}
    for index in range(8):
        width=1408 if index%2 else 1024
        image=np.empty((1024,width,3),dtype=np.uint8)
        image[:,:,0]=index*29
        image[:,:,1]=np.arange(1024,dtype=np.int64)[:,None]%251
        image[:,:,2]=np.arange(width,dtype=np.int64)[None,:]%251
        boxes=np.array([[0.,0.,200.,200.],[400.,300.,700.,900.],[900.,20.,float(width),400.]])
        records[index]=(image,boxes,'spike' if index%2 else 'primary')
    snapshots={i:(a.copy(),b.copy()) for i,(a,b,_) in records.items()}
    for seed in range(64):
        actual_rng,reference_rng=random.Random(seed),random.Random(seed)
        actual_calls,reference_calls=[],[]
        def load(key,calls):
            calls.append(key)
            image,boxes,role=records[key]
            return image,np.empty((0,4)) if seed==0 else boxes,role
        observed=mosaic_image(seed%8,list(records),lambda key:load(key,actual_calls),python_rng=actual_rng)
        namespace=dict(np=np,random=reference_rng)
        exec(code,namespace)
        owner=SimpleNamespace(image_ids=list(records),bbox_removal_threshold=.25,
            load_image_and_boxes=lambda key:load(key,reference_calls))
        owner.crop_image=lambda *args,**kwargs:namespace['crop_image'](owner,*args,**kwargs)
        expected=namespace['load_cutmix_image_and_boxes'](owner,seed%8)
        for a,b in zip(observed,expected):np.testing.assert_array_equal(a,b)
        assert observed[1].dtype==expected[1].dtype
        assert actual_calls==reference_calls and actual_rng.getstate()==reference_rng.getstate()
    for key,(image,boxes,_) in records.items():
        np.testing.assert_array_equal(image,snapshots[key][0]);np.testing.assert_array_equal(boxes,snapshots[key][1])
    files=['sciona/wheat_mosaic.py','sciona/wheat_crop.py','scripts/validate_wheat_mosaic.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=digest,
        exact_cases=64,output_size=1024,pixels_boxes_loader_order_and_rng_exact=True,
        ordinary_wide_auxiliary_and_empty_boxes_exercised=True,inputs_unchanged=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Mosaic assembly only; single-image sampling, resizing, photometric augmentation and full fits remain pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_cases=64)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.output)
