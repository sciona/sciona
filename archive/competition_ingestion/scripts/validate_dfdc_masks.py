"""Synthetic source arithmetic and independent oracles for training masks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_difference_mask import difference_mask
from sciona.dfdc_occlusion_masks import prepare_bit_masks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    for name in ('preprocessing/generate_diffs.py','training/datasets/classifier_dataset.py'):
        assert hashlib.sha256((args.source_root/name).read_bytes()).hexdigest()==next(f['sha256'] for f in pins['files'] if f['path']==name)
    tree=ast.parse((args.source_root/'preprocessing/generate_diffs.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save_diffs')
    block=next(n for n in ast.walk(fn) if isinstance(n,ast.Try))
    source=compile(ast.Module(body=[block],type_ignores=[]),'<source-ssim-mask-block>','exec')
    saved=[]
    def legacy_api(a,b,*,multichannel,full):
        assert multichannel and full
        return structural_similarity(a,b,channel_axis=-1,full=True)
    namespace={'np':np,'compare_ssim':legacy_api,'diff_path':'',
               'cv2':SimpleNamespace(cvtColor=cv2.cvtColor,COLOR_BGR2GRAY=cv2.COLOR_BGR2GRAY,
                                     imwrite=lambda _,image:saved.append(image.copy()))}
    cases=0
    rng=np.random.default_rng(926)
    for shape in ((7,7,3),(17,19,3),(25,13,3)):
        a=rng.integers(0,256,shape,dtype=np.uint8)
        for b in (a.copy(),255-a,np.roll(a,1,axis=0),rng.integers(0,256,shape,dtype=np.uint8)):
            saved.clear();namespace.update(img1=a[:,:,::-1].copy(),img2=b[:,:,::-1].copy())
            exec(source,namespace)
            actual=difference_mask(a,b)
            assert len(saved)==1
            np.testing.assert_array_equal(actual,saved[0]);cases+=1
    for a,b in ((np.zeros((6,8,3),dtype=np.uint8),np.zeros((6,8,3),dtype=np.uint8)),
                (np.zeros((8,8,3),dtype=np.uint8),np.zeros((8,9,3),dtype=np.uint8))):
        saved.clear();namespace.update(img1=a,img2=b);exec(source,namespace)
        assert not saved and difference_mask(a,b) is None;cases+=1
    original=next(n for n in ast.parse((args.source_root/'training/datasets/classifier_dataset.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_bit_masks')
    current=next(n for n in ast.parse((ROOT/'sciona/dfdc_occlusion_masks.py').read_text()).body if isinstance(n,ast.FunctionDef))
    assert ast.dump(original)==ast.dump(current)
    namespace={'np':np};exec(compile(ast.Module(body=[original],type_ignores=[]),'<source-region-masks>','exec'),namespace)
    patterns=0
    for h,w in ((8,8),(5,10),(10,5),(7,9)):
        mask=rng.integers(0,2,(h,w),dtype=np.uint8)
        actual=prepare_bit_masks(mask);expected=namespace['prepare_bit_masks'](mask)
        rows,cols=np.indices((h,w));row_half=rows>=w//2;col_half=cols>=w//2
        independent=[row_half,~row_half,col_half,~col_half,row_half!=col_half,row_half==col_half]
        for a,b,c in zip(actual,expected,independent):
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c.astype(np.uint8));patterns+=1
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_dfdc_difference_mask.py'],cwd=ROOT,check=True)
    files=['sciona/dfdc_difference_mask.py','sciona/dfdc_occlusion_masks.py',
           'tests/test_dfdc_difference_mask.py','scripts/validate_dfdc_masks.py',
           'docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-training-masks-validation.v1','result':'passed','source_commit':pins['commit'],
        'dependency':{'scikit_image':skimage.__version__},
        'checks':{'source_difference_mask_cases':cases,'independent_ssim_tests':6,
                  'region_mask_ast_equal':True,'source_and_independent_region_patterns':patterns},
        'semantics':['RGB input explicitly reordered to source BGR before per-channel SSIM.',
                     'Difference float maps cast to uint8 before grayscale; above255 wrapping retained.',
                     'Uncomputable small/mismatched pairs yield missing mask; training zero-mask substitution remains to connect.',
                     'Six region patterns use source width-derived row midpoint, including rectangular-image behavior.'],
        'limits':'Modern structural_similarity replaces removed compare_ssim API in implementation and source harness. Independent constant/negative-SSIM oracles passed, but historical binary equivalence is not asserted. Synthetic crop pairs only; no original media, identities, full preparation or promotion claim.',
        'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_training_masks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
