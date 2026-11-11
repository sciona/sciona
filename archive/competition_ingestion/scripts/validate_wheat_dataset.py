"""Compare complete runtime-array dataset sampling with the historical class."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from sciona.wheat_augmentation_compat import historical_augmentation_api
from sciona.wheat_dataset import WheatDataset


def main(source,output):
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!='8224574f23ae68f559e7f9a5a33bbb9e207dfd5ed76f6145db16dfb38c02c84c':raise ValueError('historical dataset source drift')
    tree=ast.parse(raw)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in {'get_aug','bb_overlap','WheatDataset'}]
    code=compile(ast.Module(body=nodes,type_ignores=[]),'<historical-dataset>','exec')
    records={}
    for index in range(8):
        width=1408 if index%2 else 1024
        image=np.empty((1024,width,3),dtype=np.uint8)
        image[:,:,0]=index*29;image[:,:,1]=np.arange(1024)[:,None]%251;image[:,:,2]=np.arange(width)[None,:]%251
        boxes=np.array([[20.,20.,900.,900.],[0.,0.,9.,20.],[100.,100.,110.,110.]])
        if index==0:boxes=np.empty((0,4))
        records[index]=(image,boxes,'spike' if index%2 else 'primary')
    cases=0;retry_cases=0
    with historical_augmentation_api():
        import albumentations as A
        import imgaug
        import cv2
        cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
        namespace=dict(vars(A),np=np,random=random,torch=torch,Dataset=torch.utils.data.Dataset)
        exec(code,namespace)
        for size in (512,640,768,1024):
            for network in ('FasterRCNN','EffDet'):
                for mode in ('train','valid'):
                    for seed in range(8):
                        actual_calls=[];reference_calls=[]
                        def loader(key,calls):
                            calls.append(int(key));image,boxes,role=records[key]
                            return image.copy(),boxes.copy(),role
                        random.seed(seed);np.random.seed(seed);imgaug.seed(seed)
                        actual=WheatDataset(list(records),lambda key:loader(key,actual_calls),size,mode,network)
                        # Validation excludes the wide auxiliary population in the source.
                        index=0 if mode=='train' else actual.image_ids.index(0 if seed==0 else 2)
                        observed=actual[index]
                        state=(random.getstate(),np.random.get_state(),imgaug.current_random_state().get_state())
                        random.seed(seed);np.random.seed(seed);imgaug.seed(seed)
                        reference=namespace['WheatDataset'](pd.DataFrame({'image_id':list(records)}),size,mode,network)
                        def reference_load(key):
                            image,boxes,role=loader(key,reference_calls)
                            boxes=reference.refine_boxes(boxes)
                            return image,np.array(boxes,dtype=float).reshape(-1,4),role
                        reference.load_image_and_boxes=reference_load
                        expected=reference[index]
                        assert actual.image_ids==reference.image_ids and actual_calls==reference_calls
                        torch.testing.assert_close(observed[0],expected[0],rtol=0,atol=0)
                        assert set(observed[1])==set(expected[1])
                        for key in observed[1]:torch.testing.assert_close(observed[1][key],expected[1][key],rtol=0,atol=0)
                        assert state[0]==random.getstate()
                        for a,b in zip(state[1:],(np.random.get_state(),imgaug.current_random_state().get_state())):
                            assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
                        retry_cases+=int(len(actual_calls)>4)
                        cases+=1
    files=['sciona/wheat_dataset.py','sciona/wheat_augmentation.py','sciona/wheat_augmentation_compat.py',
           'sciona/wheat_crop.py','sciona/wheat_mosaic.py','scripts/validate_wheat_dataset.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=digest,
        exact_dataset_cases=cases,cases_with_more_than_four_loads=retry_cases,
        image_tensors_targets_load_order_and_rng_exact=True,all_four_sizes_both_networks_and_modes=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Runtime-array loader replaces file decoding; real auxiliary data and historical image codecs are not qualified.',
                'Pinned historical augmentation packages use installed native dependencies; complete base fits remain pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_dataset_cases=cases,retry_cases=retry_cases)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.output)
