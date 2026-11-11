"""Source pseudo selection/append parity and unchanged retraining loop audit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.hubmap_retraining import append_pseudo,prepare_retraining
from sciona.hubmap_raw_training import prepare_fold
from sciona.hubmap_raw_population import prepare_population


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    for name,digest in pins['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    path=source/'src/05_train_with_pseudo_labels/train_05.py'
    nodes=[]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.Assign):
            text=ast.unparse(node)
            if text.startswith("pseudo_df['binned'] =") or text.startswith("pseudo_df['is_masked'] =") or text.startswith("pseudo_df = pseudo_df[pseudo_df['is_masked'] == True]"):
                nodes.append(node)
    assert len(nodes)==3
    selection=compile(ast.Module(body=sorted(nodes,key=lambda n:n.lineno),type_ignores=[]),'<original-pseudo-selection>','exec')
    tree=ast.parse((source/'src/05_train_with_pseudo_labels/run.py').read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='pseudo_df is not None']
    assert len(nodes)==1
    append=compile(ast.Module(body=nodes,type_ignores=[]),'<original-pseudo-append>','exec')
    # Exact AST equality means existing independent training-loop comparisons
    # also cover this loop, with its supplied pseudo-augmented population.
    equivalent=[]
    for name in ['dataset.py','run.py']:
        a=ast.dump(ast.parse((source/'src/02_train'/name).read_text()),include_attributes=False)
        b=ast.dump(ast.parse((source/'src/05_train_with_pseudo_labels'/name).read_text()),include_attributes=False)
        assert a==b
        equivalent.append(name)
    def recipe(path):
        return [ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                if not isinstance(n,(ast.Import,ast.ImportFrom))]
    assert recipe(source/'src/02_train/transforms.py')==recipe(source/'src/05_train_with_pseudo_labels/transforms.py')
    cases=0;discarded=0
    for size in [16,32]:
        slides=[]
        for i in range(3):
            image=np.random.RandomState(347+i).randint(0,256,(size+3,size+5,3)).astype(np.uint8)
            mask=np.zeros(image.shape[:2],dtype=np.uint8)
            if i!=1:mask[::3]=1
            slides.append((image,mask))
        groups=['synthetic-a','synthetic-b','synthetic-c']
        training,validation=prepare_fold(slides,groups,['synthetic-c'],tile_size=size)
        def frame(pop):
            return pd.DataFrame(dict(image=pop['images_bgr'],rle=pop['rles'],binned=pop['bins'],is_masked=pop['present']))
        for kinds in [[],['empty'],['sparse'],['positive'],['sparse','positive'],['positive','positive']]:
            sources=[];populations=[];ratios=[]
            for index,kind in enumerate(kinds):
                image=np.random.RandomState(353+index).randint(0,256,(size+3,size+5,3)).astype(np.uint8)
                mask=np.zeros(image.shape[:2],dtype=np.uint8)
                if kind=='sparse':mask[0,0]=1
                if kind=='positive':mask[::2]=1
                sources.append([(image,mask)])
                pop=prepare_population(sources[-1],tile_size=size)
                populations.append(pop)
                # Recover exact areas from canonical synthetic RLE for original
                # binned recomputation; no real labels or records are involved.
                for rle in pop['rles']:
                    runs=[int(x) for x in rle.split()]
                    ratios.append(sum(runs[1::2])/(size*size))
            pseudo=pd.concat([frame(p) for p in populations],ignore_index=True) if populations else pd.DataFrame(columns=['image','rle','binned','is_masked'])
            pseudo['ratio_masked_area']=ratios
            ns=dict(np=np,pd=pd,pseudo_df=pseudo,config=dict(multiplier_bin=20),trn_df=frame(training))
            exec(selection,ns);discarded+=len(pseudo)-len(ns['pseudo_df'])
            exec(append,ns);expected=ns['trn_df']
            actual,valid=prepare_retraining(slides,groups,['synthetic-c'],sources,tile_size=size)
            assert actual['rles']==expected['rle'].tolist()
            for x,y in zip(actual['images_bgr'],expected['image']):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(actual['bins'],expected['binned'].to_numpy(dtype=int))
            np.testing.assert_array_equal(actual['present'],expected['is_masked'].to_numpy(dtype=bool))
            assert valid['rles']==validation['rles']
            for x,y in zip(valid['images_bgr'],validation['images_bgr']):np.testing.assert_array_equal(x,y)
            cases+=1
    paths=['sciona/hubmap_retraining.py','sciona/hubmap_raw_training.py','sciona/hubmap_raw_population.py',
           'sciona/hubmap_preparation.py','sciona/hubmap_tiling.py','scripts/validate_hubmap_retraining.py',
           'docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False,synthetic_only=True,exact_population_and_validation_cases=cases,
                discarded_pseudo_tiles=discarded,source_ast_identical=equivalent,augmentation_recipe_ast_identical=True,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Original pseudo bin/presence filter and training append executed; raw tile numerics have separate source evidence.',
                             'Runtime ordered pseudo source collections; original private identities not embedded.',
                             'Full pseudo-augmented training execution, pseudo preparation orchestration and final graph publication remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_retraining.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
