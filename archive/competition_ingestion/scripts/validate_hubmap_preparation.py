"""Compare in-memory preparation with original synthetic JPEG/pickle file IO."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import cv2
import numpy as np
import pandas as pd
from sciona.hubmap_preparation import prepare_tile,select_training_tiles


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    extra=json.loads((root/'docs/reviews/competition_hubmap_loader_pins.json').read_text())
    ns=dict(np=np,cv2=cv2,pickle=pickle,opj=lambda a,b:str(Path(a)/b))
    for path,name in [('src/utils.py','mask2rle'),('src/utils_data_generation.py','generate_data')]:
        assert hashlib.sha256((source/path).read_bytes()).hexdigest()=={**pins['files'],**extra['files']}[path]
        nodes=[n for n in ast.parse((source/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name]
        assert len(nodes)==1
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-tile-preparation>','exec'),ns)
    path='src/02_train/train_02.py'
    assert hashlib.sha256((source/path).read_bytes()).hexdigest()==pins['files'][path]
    selected=[]
    for node in ast.walk(ast.parse((source/path).read_text())):
        if isinstance(node,ast.Assign):
            text=ast.unparse(node)
            if text.startswith("data_df = data_df[data_df['std_img'] > 10]") or text.startswith("data_df['binned'] =") or text.startswith("data_df['is_masked'] ="):
                selected.append(node)
    assert len(selected)==3
    selection=compile(ast.Module(body=sorted(selected,key=lambda n:n.lineno),type_ignores=[]),'<pinned-tile-selection>','exec')
    records=[];cases=0;jpeg_changes=0
    with tempfile.TemporaryDirectory(prefix='sciona-hubmap-synthetic-') as directory:
        for side in [32,64,1024]:
            shapes=['random','flat','boundary','above'] if side!=1024 else ['random']
            for kind in shapes:
                image=np.random.RandomState(227).randint(0,256,(side,side,3)).astype(np.uint8)
                if kind=='flat':image.fill(0)
                if kind in ['boundary','above']:
                    image[:]=((np.indices((side,side)).sum(0)%2)*(20 if kind=='boundary' else 22))[...,None]
                for foreground in ['empty','sparse','full']:
                    mask=np.zeros((side,side),dtype=np.uint8)
                    if foreground=='sparse':mask[0,0]=1
                    if foreground=='full':mask.fill(1)
                    actual=prepare_tile(image,mask)
                    data=ns['generate_data']('synthetic',0,image.copy(),mask[...,None].copy(),dict(OUTPUT_PATH=directory))
                    decoded=cv2.imread(str(Path(directory)/data[0]))
                    with (Path(directory)/data[1]).open('rb') as f:rle=pickle.load(f)
                    np.testing.assert_array_equal(actual['image_bgr'],decoded)
                    assert actual['rle']==rle
                    assert [actual['num_masked_pixels'],actual['ratio_masked_area'],actual['std_img']]==data[2:]
                    records.append(actual);cases+=1
                    jpeg_changes+=int(not np.array_equal(decoded,image[:,:,::-1]))
    frame=pd.DataFrame(dict(std_img=[r['std_img'] for r in records],ratio_masked_area=[r['ratio_masked_area'] for r in records],_position=np.arange(len(records))))
    env=dict(np=np,data_df=frame,config=dict(multiplier_bin=20))
    exec(selection,env);expected=env['data_df']
    result=select_training_tiles(records,multiplier_bin=20)
    np.testing.assert_array_equal(result['bins'],expected['binned'].to_numpy())
    np.testing.assert_array_equal(result['present'],expected['is_masked'].to_numpy())
    for actual,index in zip(result['images_bgr'],expected['_position']):np.testing.assert_array_equal(actual,records[index]['image_bgr'])
    assert result['rles']==[records[i]['rle'] for i in expected['_position']]
    paths=['sciona/hubmap_preparation.py','scripts/validate_hubmap_preparation.py','tests/test_hubmap_preparation.py',
           'docs/reviews/competition_hubmap_source_pins.json','docs/reviews/competition_hubmap_loader_pins.json']
    return dict(approved=False,synthetic_only=True,exact_preparation_cases=cases,lossy_jpeg_cases=jpeg_changes,
        selected_tiles=len(expected),exact_selection_bins_presence=True,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Current OpenCV JPEG defaults match original file IO under the same backend; historical codec binaries excluded.',
            'Temporary synthetic files only; no real data or identifying metadata retained.',
            'Preparation and selection components; full raw-input training and inference graph remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
