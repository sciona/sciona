"""Compare cropping and inclusive source box-coverage retention."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sciona.wheat_crop import crop_image_and_boxes


def main(source,output):
    raw=source.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!='8224574f23ae68f559e7f9a5a33bbb9e207dfd5ed76f6145db16dfb38c02c84c':
        raise ValueError('historical crop source drift')
    tree=ast.parse(raw)
    overlap=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bb_overlap')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WheatDataset')
    crop=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='crop_image')
    namespace=dict(np=np)
    exec(compile(ast.Module(body=[overlap,crop],type_ignores=[]),'<historical-crop>','exec'),namespace)
    rng=np.random.default_rng(1381)
    image=rng.integers(0,256,(1024,1024,3),dtype=np.uint8)
    cases=0
    for index in range(128):
        starts=rng.integers(0,900,(31,2))
        boxes=np.concatenate([starts,starts+rng.integers(1,124,(31,2))],axis=1).astype(float)
        xmin,ymin=map(int,rng.integers(0,400,2));xmax,ymax=map(int,rng.integers(600,1025,2))
        if index==0:boxes=np.empty((0,4))
        if index==1:
            xmin,ymin,xmax,ymax=0,0,10,10
            boxes=np.array([[7.,0.,22.,3.],[6.99,0.,22.,3.],[7.01,0.,22.,3.]])
        before=boxes.copy()
        observed=crop_image_and_boxes(image,boxes,xmin,ymin,xmax,ymax)
        expected=namespace['crop_image'](SimpleNamespace(bbox_removal_threshold=.25),image,boxes,xmin,ymin,xmax,ymax)
        for a,b in zip(observed,expected):np.testing.assert_array_equal(a,b)
        assert observed[1].dtype==expected[1].dtype
        np.testing.assert_array_equal(boxes,before)
        if index==1:assert len(observed[1])==1
        cases+=1
    files=['sciona/wheat_crop.py','scripts/validate_wheat_crop.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=digest,
        exact_cases=cases,strict_quarter_coverage_boundary_exercised=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Crop geometry only; crop sampling, mosaic assembly and photometric augmentation remain separate.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_cases=cases)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.output)
