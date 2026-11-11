"""Pinned training mask/crop and voxel annotation comparison on synthetic arrays."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
from scipy.ndimage import zoom,binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_training_preprocessing import prepare_training_masks,training_voxel_annotations
from sciona.dsb_world_preprocessing import world_training_annotations,prepare_world_training_volume


def validate(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    tree=ast.parse(checked_source(source_root,pins,'training/prepare.py'))
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['process_mask','lumTrans','resample','worldToVoxelCoord']]
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='savenpy')
    begin=next(i for i,n in enumerate(function.body) if ast.unparse(n)=='Mask = m1 + m2')
    end=next(i for i,n in enumerate(function.body) if ast.unparse(n).startswith('np.save('))
    body=function.body[begin:end]
    for i,n in enumerate(body):
        if ast.unparse(n)=='extramask = dilatedMask - Mask':
            body[i]=ast.parse('extramask = dilatedMask.astype(int) - Mask.astype(int)').body[0]
    maskcode=compile(ast.Module(body=body,type_ignores=[]),'<source-training-cleanup>','exec')
    annotation=next(n for n in function.body if isinstance(n,ast.If) and ast.unparse(n.test)=='len(label) == 0')
    labelcode=compile(ast.Module(body=[annotation],type_ignores=[]),'<source-voxel-annotations>','exec')
    base=dict(np=np,zoom=zoom,binary_dilation=binary_dilation,generate_binary_structure=generate_binary_structure,
              convex_hull_image=convex_hull_image)
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'<source-preparation-helpers>','exec'),base)
    count=0
    for dtype in [np.float32,np.float64]:
        for seed in range(3):
            rng=np.random.RandomState(seed)
            volume=rng.uniform(-1300,700,(12,40,40)).astype(dtype);volume[0,0,0]=np.nan
            left=np.zeros(volume.shape,dtype=bool);left[3:9,8:32,5:13]=True
            right=np.zeros_like(left);right[3:9,8:32,25:33]=True
            spacing=np.array([1.7,1.2,.9]);resolution=np.array([1.,1.,1.])
            scope={**base,'im':volume.copy(),'m1':left,'m2':right,'spacing':spacing,'resolution':resolution}
            exec(maskcode,scope)
            actual,effective,bounds=prepare_training_masks(volume,left,right,spacing,resolution)
            np.testing.assert_array_equal(actual,scope['sliceim'])
            np.testing.assert_array_equal(bounds,scope['extendbox'])
            for rows in [np.array([[11.,17,5,8],[22,20,6,10]]),np.empty((0,4)),np.zeros((1,4))]:
                scope['label']=rows[:,[2,0,1,3]].copy()
                exec(labelcode,scope)
                expected=scope['label2']
                if np.all(expected==0):expected=np.empty((0,4))
                np.testing.assert_array_equal(training_voxel_annotations(rows,spacing,bounds,resolution),expected)
            count+=1
    auxiliary=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='savenpy_luna')
    label_branch=next(n for n in auxiliary.body if isinstance(n,ast.If) and ast.unparse(n.test)=='islabel')
    # Omit only row selection by source identity and the output save call.
    worldcode=compile(ast.Module(body=label_branch.body[1:-1],type_ignores=[]),'<source-world-annotations>','exec')
    world_cases=0
    for seed in range(3):
        rng=np.random.RandomState(seed)
        for flip in [False,True]:
            for size in [0,1,5]:
                rows=np.column_stack((rng.uniform(-20,50,(size,3)),rng.uniform(1,12,size)))
                scope={**base,'this_annos':np.column_stack((np.zeros(size),rows)),
                    'origin':np.array([10.,-3.,12.]),'spacing':np.array([1.7,1.2,.9]),
                    'resolution':np.array([1.,.8,1.1]),'extendbox':bounds,
                    'Mask':np.empty(volume.shape,dtype=bool),'isflip':flip}
                exec(worldcode,scope)
                expected=scope['label2'] if size else np.empty((0,4))
                actual=world_training_annotations(rows,scope['origin'],scope['spacing'],bounds,volume.shape,
                    flip=flip,requested_spacing=scope['resolution'])
                np.testing.assert_array_equal(actual,expected)
                world_cases+=1
    class NoPrint(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print':
                return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
    auxcode=compile(ast.fix_missing_locations(NoPrint().visit(ast.Module(body=[auxiliary],type_ignores=[]))),
                    '<source-auxiliary-array-io>','exec')
    auxiliary_cases=0
    for dtype in [np.float32,np.float64]:
        for flip in [False,True]:
            saved=[];loads=[]
            image=np.linspace(-1300,700,12*40*40).reshape(12,40,40).astype(dtype)
            segmentation=left.astype(np.int16)*3+right.astype(np.int16)*4
            def load_image(_):
                loads.append(True)
                return ((segmentation if len(loads)==1 else image).copy(),np.array([10.,-3.,12.]),spacing,flip)
            proxy=SimpleNamespace(**{key:getattr(np,key) for key in dir(np) if not key.startswith('_')})
            proxy.save=lambda _,value:saved.append(value.copy())
            scope={**base,'np':proxy,'load_itk_image':load_image,
                   'os':SimpleNamespace(path=SimpleNamespace(join=lambda *args:'synthetic'))}
            exec(auxcode,scope)
            rows=np.array([[11.,17.,5.,8.]])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                scope['savenpy_luna'](0,np.column_stack(([0],rows)),['0'],'synthetic','synthetic','synthetic')
            actual=prepare_world_training_volume(image,left,right,spacing,[10.,-3.,12.],rows,flip=flip)
            assert len(saved)==2 and len(loads)==2
            np.testing.assert_array_equal(actual[0],saved[0]);np.testing.assert_array_equal(actual[1],saved[1])
            auxiliary_cases+=1
    paths=['sciona/dsb_training_preprocessing.py','tests/test_dsb_training_preprocessing.py',
           'sciona/dsb_world_preprocessing.py','tests/test_dsb_world_preprocessing.py',
           'scripts/validate_dsb_training_preprocessing.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],exact_preprocessing_cases=count,
        exact_annotation_cases=count*3,exact_world_annotation_cases=world_cases,exact_auxiliary_preparation_cases=auxiliary_cases,
        adaptations=['Boolean subtraction explicitly converted to integer subtraction in reference.',
            'Array IO and source identifier lookup excluded; empty source sentinel converted to loader-ready empty array.'],
        limitations=['Source image IO/orientation extraction not covered; caller supplies explicit orientation.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_training_preprocessing.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
