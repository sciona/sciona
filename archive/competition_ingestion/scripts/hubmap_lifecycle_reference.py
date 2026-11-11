"""Independent source execution helpers for connected HuBMAP lifecycle checks."""
import ast
import hashlib
import json
from pathlib import Path
import pickle
import io
import random
import contextlib
import tempfile
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import torch
from scripts.validate_hubmap_loader import encode_synthetic
from scripts.validate_hubmap_loader import source_dataset
from scripts.validate_hubmap_network import source_model
from scripts.validate_hubmap_validation import source_validation
from scripts.validate_hubmap_epoch import source_training
from scripts.validate_hubmap_sampling_schedule import source_functions
from scripts.validate_hubmap_checkpoint_policy import reference as source_policy
from scripts.validate_hubmap_inference_tiles import source_tiles
from scripts.validate_hubmap_final_network import notebook_model,notebook_nodes


def infer(root,source,library,notebook,payload,image,python_seed,torch_seed,final=False):
    if final:
        model=notebook_model(root,source,notebook)((32,32),False,False,load_weights=False).eval()
    else:model=source_model(root,source)((32,32),False,False,.5,load_weights=False).eval()
    model.load_state_dict(torch.load(io.BytesIO(payload),weights_only=True))
    dataset=source_tiles(root,source,library,image,resolution=32,input_resolution=32,pad_size=8)
    if final:
        definitions=notebook_nodes(root,notebook)
        namespace=dict(dataset.__class__.__init__.__globals__);namespace['INPUT_PATH']='synthetic'
        constants=[]
        for cell in json.loads(notebook.read_text())['cells']:
            if cell['cell_type']!='code':continue
            try:tree=ast.parse(''.join(cell['source']))
            except SyntaxError:continue
            constants.extend(n for n in tree.body if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id in {'MEAN','STD'} for t in n.targets))
        assert len(constants)==2
        exec(compile(ast.Module(body=constants+[definitions[n] for n in ['get_transforms_test','HuBMAPDataset']],type_ignores=[]),
                     '<final-notebook-dataset>','exec'),namespace)
        dataset=namespace['HuBMAPDataset'](0,pd.DataFrame(dict(id=['synthetic'])))
        nodes=[definitions[n] for n in ['my_collate_fn','get_pred_mask']]
    else:
        path=source/'src/03_generate_pseudo_labels/03_01_pseudo_label_kaggle_data/utils_inference.py'
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'my_collate_fn','get_pred_mask'}]
    assert len(nodes)==2
    generator=torch.Generator().manual_seed(torch_seed)
    def loader(*args,**kwargs):return torch.utils.data.DataLoader(*args,**kwargs,generator=generator)
    ns=dict(np=np,cv2=cv2,torch=torch,DataLoader=loader,HuBMAPDataset=lambda *args:dataset,tqdm=lambda x:x,
            seed=0,device=torch.device('cpu'),config=dict(test_batch_size=4,tta=3 if final else 4,mask_threshold=.5))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-lifecycle-inference>','exec'),ns)
    saved=random.getstate();random.seed(python_seed)
    try:
        with contextlib.redirect_stdout(io.StringIO()):mask,h,w=ns['get_pred_mask'](0,None,[[model]])
        return dict(mask=mask,python_state=random.getstate(),torch_state=generator.get_state())
    finally:random.setstate(saved)


def train(root,source,library,frame,encoder_state,seed,pseudo=None):
    original=source_model(root,source)
    namespace=original.__init__.__globals__['pretrainedmodels']
    factory=namespace.se_resnext101_32x4d
    def restored_encoder(**kwargs):
        encoder=factory(**kwargs)
        encoder.load_state_dict(encoder_state)
        return encoder
    namespace.se_resnext101_32x4d=restored_encoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=original((32,32),True,True,None,load_weights=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-5)
    sample,schedule=source_functions(root,source)
    scheduler=schedule(optimizer,step_size_min=1e-6,t0=19,tmult=1)
    partition_pins=json.loads((root/'docs/reviews/competition_hubmap_partition_pins.json').read_text())
    path='src/get_fold_idxs_list.py'
    assert hashlib.sha256((source/path).read_bytes()).hexdigest()==partition_pins['files'][path]
    nodes=[n for n in ast.parse((source/path).read_text()).body if isinstance(n,ast.FunctionDef)]
    ns=dict(np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-folds>','exec'),ns)
    ti,vi=ns['get_fold_idxs_list'](frame,[['synthetic-valid']])
    training=frame.iloc[ti[0]].copy();validation=frame.iloc[vi[0]].copy()
    if pseudo is not None:
        tree=ast.parse((source/'src/05_train_with_pseudo_labels/train_05.py').read_text())
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
               ast.unparse(n).startswith("pseudo_df = pseudo_df[pseudo_df['is_masked'] == True]")]
        assert len(nodes)==1
        ns=dict(pseudo_df=pseudo)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-positive-pseudo>','exec'),ns)
        tree=ast.parse((source/'src/05_train_with_pseudo_labels/run.py').read_text())
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='pseudo_df is not None']
        assert len(nodes)==1
        ns.update(trn_df=training,pd=pd)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-append>','exec'),ns)
        training=ns['trn_df']
    training=training.reset_index(drop=True);training['_position']=np.arange(len(training))
    saved_py=random.getstate();saved_np=np.random.get_state()
    random.seed(seed);np.random.seed(seed+1);generator=torch.Generator().manual_seed(seed+2)
    try:
        for _ in range(17):scheduler.step()
        order=sample(training,dict(binned_max=4))
        selected=training.iloc[order]
        ds=source_dataset(root,source,library,selected['image_bgr'].tolist(),selected['rle'].tolist(),32,True)
        batches=list(torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,drop_last=True,num_workers=0,generator=generator))
        val=source_validation(root,source)
        run=source_training(root,source,val)
        training_metrics=run(model,optimizer,batches,len(order),.5)
        ds=source_dataset(root,source,library,validation['image_bgr'].tolist(),validation['rle'].tolist(),32,False)
        batches=list(torch.utils.data.DataLoader(ds,batch_size=2,shuffle=False,num_workers=0,generator=generator))
        validation_metrics=val(model,batches,len(validation),.5)
        policy=source_policy(root,source)
        decisions,_=policy([(18,validation_metrics['loss'],validation_metrics['dice'])],
            dict(early_stopping=True,patience=10,lr_scheduler_name='CosineAnnealingLR',lr_scheduler={'CosineAnnealingLR':dict(t0=19)}))
        assert 'best_loss' in decisions[0]['save']
        if decisions[0]['step_scheduler']:scheduler.step()
        stream=io.BytesIO();torch.save(model.state_dict(),stream)
        return dict(checkpoint=stream.getvalue(),epochs=[dict(epoch=18,training=training_metrics,
                    validation=validation_metrics,decision=decisions[0])],
                    python_state=random.getstate(),numpy_state=np.random.get_state(),torch_state=generator.get_state())
    finally:
        random.setstate(saved_py);np.random.set_state(saved_np)


def prepare(root,source,slides,groups,size=32):
    pins = {}
    for name in ['source', 'loader', 'pseudo', 'partition']:
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
    records = []
    ties = 0
    with tempfile.TemporaryDirectory(prefix='sciona-synthetic-raw-') as directory:
        for grid, shift in enumerate([0,size//2]):
            for slide_index, (image, mask) in enumerate(slides):
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
                    records.append(dict(image=cv2.imread(str(Path(directory)/row[0])),rle=rle,std_img=row[4],ratio_masked_area=row[3],patient_number=groups[slide_index]))
    frame=pd.DataFrame(records)
    env=dict(np=np,data_df=frame,config=dict(multiplier_bin=20))
    exec(selection,env)
    result=env['data_df'].copy()
    result['image_bgr']=result.pop('image')
    return result.reset_index(drop=True)
