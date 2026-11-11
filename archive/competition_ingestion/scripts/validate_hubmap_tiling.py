"""Execute pinned raw tiler class with synthetic array-backed raster windows."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sciona.hubmap_tiling import RawTiles
from scripts.validate_hubmap_loader import encode_synthetic


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    extra=json.loads((root/'docs/reviews/competition_hubmap_loader_pins.json').read_text())
    for name,digest in {**pins['files'],**extra['files']}.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    functions=[n for n in ast.parse((source/'src/utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='rle2mask']
    classes=[n for n in ast.parse((source/'src/utils_data_generation.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HuBMAPDataset']
    assert len(functions)==len(classes)==1
    reports=[]
    for h,w,size in [(32,32,16),(31,33,16),(9,7,16),(1024,1024,1024),(1025,1031,1024)]:
        image=np.random.RandomState(223).randint(0,256,(h,w,3)).astype(np.uint8)
        mask=(np.indices((h,w)).sum(0)%5==0).astype(np.uint8)
        def read(channels,window):
            (y0,y1),(x0,x1)=window
            return np.moveaxis(image[y0:y1,x0:x1],-1,0)
        raster=SimpleNamespace(count=3,height=h,width=w,read=read)
        ns=dict(np=np,Dataset=torch.utils.data.Dataset,opj=lambda *a:'synthetic',
                rasterio=SimpleNamespace(open=lambda p:raster),Window=SimpleNamespace(from_slices=lambda y,x:(y,x)))
        exec(compile(ast.Module(body=functions+classes,type_ignores=[]),'<pinned-raw-tiler>','exec'),ns)
        for shift in [0,size//2]:
            reference=ns['HuBMAPDataset'](pd.DataFrame(dict(id=['synthetic'],encoding=[encode_synthetic(mask)])),'synthetic',
                dict(INPUT_PATH='',tile_size=size,shift_h=shift,shift_w=shift))
            actual=RawTiles(image,mask,tile_size=size,shift_h=shift,shift_w=shift)
            assert len(actual)==len(reference)
            empty=0
            for index in range(len(actual)):
                a,b=actual[index];expected=reference[index]
                np.testing.assert_array_equal(a,expected['img']);np.testing.assert_array_equal(b,expected['mask'])
                empty+=int(not a.any())
            reports.append(dict(shape=[h,w],tile_size=size,shift=shift,tiles=len(actual),fully_padded_tiles=empty,exact=True))
    paths=['sciona/hubmap_tiling.py','scripts/validate_hubmap_tiling.py','scripts/validate_hubmap_loader.py',
        'docs/reviews/competition_hubmap_source_pins.json','docs/reviews/competition_hubmap_loader_pins.json']
    return dict(approved=False,synthetic_only=True,cases=reports,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['RGB array-backed raster windows; file/subdataset codec behavior not claimed.',
            'Raw tile arrays only: JPEG storage roundtrip, statistics/binning and pseudo-label preparation remain.',
            'Original extra fully padded tiles preserved; no fabricated boundary cleanup or CDG approval.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_tiling.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(approved=False,cases=report['cases'])))
