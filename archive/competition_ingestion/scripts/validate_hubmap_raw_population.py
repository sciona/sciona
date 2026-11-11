"""Compare ordered raw preparation with pinned source on synthetic slides."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import signal
import tempfile
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import torch
from sciona.hubmap_raw_population import prepare_population
from scripts.validate_hubmap_loader import encode_synthetic


def validate(root, source):
    pins = {}
    for name in ['source', 'loader']:
        pins.update(json.loads((root/f'docs/reviews/competition_hubmap_{name}_pins.json').read_text())['files'])
    for path, digest in pins.items():
        assert hashlib.sha256((source/path).read_bytes()).hexdigest() == digest
    definitions = []
    for path, names in [('src/utils.py', {'mask2rle', 'rle2mask'}),
                        ('src/utils_data_generation.py', {'HuBMAPDataset', 'generate_data', 'my_collate_fn'})]:
        nodes = [n for n in ast.parse((source/path).read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert len(nodes) == len(names)
        definitions.extend(nodes)
    sorts = []
    for version in ['01_01', '01_02']:
        tree = ast.parse((source/f'src/01_data_preparation/{version}/data_preparation_{version}.py').read_text())
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and (ast.unparse(n).startswith('bs, sz, sz, c =')
                      or ast.unparse(n).startswith('idxs = np.argsort(')
                      or ast.unparse(n).startswith('img_patches = img_patches[idxs]')
                      or ast.unparse(n).startswith('mask_patches = mask_patches[idxs]'))]
        assert len(nodes) == 4
        sorts.append(compile(ast.Module(body=sorted(nodes, key=lambda n:n.lineno), type_ignores=[]), '<source-sort>', 'exec'))
    tree = ast.parse((source/'src/02_train/train_02.py').read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and (ast.unparse(n).startswith("data_df = data_df[data_df['std_img'] > 10]")
                  or ast.unparse(n).startswith("data_df['binned'] =")
                  or ast.unparse(n).startswith("data_df['is_masked'] ="))]
    assert len(nodes) == 3
    selection = compile(ast.Module(body=sorted(nodes, key=lambda n:n.lineno), type_ignores=[]), '<source-selection>', 'exec')
    cases = []
    for size in [16, 32, 1024]:
        rng = np.random.RandomState(239)
        slides = []
        for h, w, kind in [(size*2, size*2, 'ties'), (size+3, size+5, 'sparse')]:
            image = rng.randint(0, 256, (h,w,3)).astype(np.uint8)
            mask = np.zeros((h,w), dtype=np.uint8)
            if kind == 'ties': mask[::4] = 1
            else: mask[0,0] = 1
            slides.append((image,mask))
        records = []
        ties = 0
        with tempfile.TemporaryDirectory(prefix='sciona-synthetic-raw-') as directory:
            for grid, shift in enumerate([0,size//2]):
                for image, mask in slides:
                    h,w = mask.shape
                    def read(channels, window):
                        (y0,y1),(x0,x1) = window
                        return np.moveaxis(image[y0:y1,x0:x1], -1, 0)
                    raster = SimpleNamespace(count=3,height=h,width=w,read=read)
                    ns = dict(np=np,cv2=cv2,pickle=pickle,Dataset=torch.utils.data.Dataset,
                              opj=lambda *args:str(Path(*args)),
                              rasterio=SimpleNamespace(open=lambda p:raster),
                              Window=SimpleNamespace(from_slices=lambda y,x:(y,x)))
                    exec(compile(ast.Module(body=definitions,type_ignores=[]), '<source-preparation>', 'exec'), ns)
                    ds = ns['HuBMAPDataset'](pd.DataFrame(dict(id=['synthetic'],encoding=[encode_synthetic(mask)])),
                                             'synthetic',dict(INPUT_PATH='',tile_size=size,shift_h=shift,shift_w=shift))
                    batches = list(torch.utils.data.DataLoader(ds,batch_size=16,num_workers=0,shuffle=False,collate_fn=ns['my_collate_fn']))
                    ns.update(img_patches=np.vstack([b['img'] for b in batches]),mask_patches=np.vstack([b['mask'] for b in batches]))
                    counts = ns['mask_patches'].reshape(len(ds),-1).sum(1)
                    ties += int(len(np.unique(counts)) < len(counts))
                    exec(sorts[grid],ns)
                    for i,(x,y) in enumerate(zip(ns['img_patches'],ns['mask_patches'])):
                        row = ns['generate_data']('synthetic',i,x,y,dict(OUTPUT_PATH=directory))
                        with (Path(directory)/row[1]).open('rb') as handle:
                            rle = pickle.load(handle)
                        records.append(dict(image=cv2.imread(str(Path(directory)/row[0])),rle=rle,std_img=row[4],ratio_masked_area=row[3]))
        frame = pd.DataFrame(records)
        env = dict(np=np,data_df=frame,config=dict(multiplier_bin=20))
        exec(selection,env)
        expected = env['data_df']
        actual = prepare_population(slides,tile_size=size)
        assert len(actual['images_bgr']) == len(expected)
        for image, reference in zip(actual['images_bgr'],expected['image']):
            np.testing.assert_array_equal(image,reference)
        assert actual['rles'] == expected['rle'].tolist()
        np.testing.assert_array_equal(actual['bins'],expected['binned'])
        np.testing.assert_array_equal(actual['present'],expected['is_masked'])
        cases.append(dict(tile_size=size,slides=len(slides),generated=len(records),selected=len(expected),grids_with_ties=ties,exact=True))
    paths = ['sciona/hubmap_raw_population.py','sciona/hubmap_tiling.py','sciona/hubmap_preparation.py',
             'scripts/validate_hubmap_raw_population.py','scripts/validate_hubmap_loader.py',
             'docs/reviews/competition_hubmap_source_pins.json','docs/reviews/competition_hubmap_loader_pins.json']
    return dict(approved=False,synthetic_only=True,cases=cases,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Decoded RGB array boundary; TIFF and raster subdataset codecs excluded.',
                             'Current NumPy sort and OpenCV JPEG backend; historical binary equivalence excluded.',
                             'Training partition, pseudo-labeling and inference remain separate gates.'])


if __name__ == '__main__':
    signal.alarm(120)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_raw_population.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(approved=False,cases=report['cases'])))
