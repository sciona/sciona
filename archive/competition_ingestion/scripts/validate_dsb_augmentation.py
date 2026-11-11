"""Pinned synthetic augmentation parity and RNG-consumption comparison."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from itertools import product

import numpy as np
from scipy.ndimage import rotate

from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_augmentation import augment_detection, augment_classifier
from sciona.dsb_detector_crop import crop_detection


def validate(root, source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    counts={}
    for kind,filename in [('detector','data_detector.py'),('classifier','data_classifier.py')]:
        source=checked_source(source_root,pins,filename)
        start=source.index('def augment(')
        source=source[start:source.index('class Crop',start)] if kind=='detector' else source[start:]
        tree=ast.parse(source)
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='augment']
        count=0
        for seed in [0,7,41]:
            for flip,rotation,swap in product([False,True],repeat=3):
                original_rng=np.random.RandomState(seed); candidate_rng=np.random.RandomState(seed)
                proxy=SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')})
                proxy.random=original_rng
                scope=dict(np=proxy,rotate=rotate)
                exec(compile(tree,'<pinned-augmentation>','exec'),scope)
                sample=np.arange(16**3,dtype=float).reshape(1,16,16,16)
                coord=np.stack(np.meshgrid(*([np.linspace(-.5,.5,4)]*3),indexing='ij'))
                target=np.array([8.,8.,8.,2.]);boxes=np.array([target,[4.,5.,6.,1.]])
                if kind=='detector':
                    expected=scope['augment'](sample.copy(),target.copy(),boxes.copy(),coord.copy(),ifflip=flip,ifrotate=rotation,ifswap=swap)
                    actual=augment_detection(sample,target,boxes,coord,flip=flip,rotate=rotation,swap=swap,rng=candidate_rng)
                else:
                    expected=scope['augment'](sample.copy(),coord.copy(),ifflip=flip,ifrotate=rotation,ifswap=swap,filling_value=160)
                    actual=augment_classifier(sample,coord,flip=flip,rotate=rotation,swap=swap,rng=candidate_rng)
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(candidate_rng.rand(10),original_rng.rand(10))
                count+=1
        counts[kind]=count
    import warnings
    from scipy.ndimage import zoom
    source=checked_source(source_root,pins,'data_detector.py')
    tree=ast.parse(source[source.index('class Crop('):source.index('class LabelMapping(')].expandtabs(8))
    changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
            text=ast.unparse(node)
            if ((isinstance(node.right,ast.Attribute) and node.right.attr=='stride')
                    or ('target' not in text and isinstance(node.right,ast.Constant) and node.right.value==2)):
                node.op=ast.FloorDiv();changed+=1
    assert changed==10
    count=0
    for seed in [0,7,41]:
        for scale,random_crop in product([False,True],repeat=2):
            for target in [np.array([24.,24.,24.,10.]),np.array([2.,3.,4.,10.])]:
                original_rng=np.random.RandomState(seed);candidate_rng=np.random.RandomState(seed)
                proxy=SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')})
                proxy.random=original_rng
                scope=dict(np=proxy,zoom=zoom,warnings=warnings)
                exec(compile(tree,'<pinned-detector-crop-python3>','exec'),scope)
                volume=np.arange(48**3,dtype=np.float32).reshape(1,48,48,48)
                boxes=np.array([target,[20.,21.,22.,4.]])
                config=dict(crop_size=[32]*3,bound_size=4,stride=4,pad_value=170)
                expected=scope['Crop'](config)(volume,target,boxes,isScale=scale,isRand=random_crop)
                actual=crop_detection(volume,target,boxes,crop_size=32,bound_size=4,scale=scale,
                                      random_crop=random_crop,rng=candidate_rng)
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(original_rng.rand(10),candidate_rng.rand(10))
                count+=1
    counts['detector_crop']=count
    paths=['sciona/dsb_augmentation.py','sciona/dsb_detector_crop.py','tests/test_dsb_augmentation.py','scripts/validate_dsb_augmentation.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],exact_array_and_rng_parity_cases=counts,
        limitations=['Source size-coordinate detector flip convention retained.',
                     'Source classifier rotation does not rotate coordinates; disabled in source defaults.',
                     'Same current SciPy/NumPy, not historic library behavior proof.',
                     'Detector crop reference ports ten Python 2 integer quotients.',
                     'Training sampling sequence and optimizer integration remain.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_augmentation_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
