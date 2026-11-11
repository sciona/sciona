"""Execute pinned validation getitem methods on synthetic arrays and explicit RNGs."""
import argparse
import ast
import hashlib
import json
import random
import warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.ndimage import zoom,rotate,binary_dilation,generate_binary_structure
from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_intake import classifier_validation_sample,detector_validation_sample


def reference(source,role,rng,label_rng,volume):
    tree=ast.parse(source.expandtabs(8))
    name='DataBowl3Classifier' if role=='classifier' else 'DataBowl3Detector'
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    # The first two statements read time and reseed. All remaining getitem code
    # executes unchanged apart from Python 2 integer dimensions.
    assert ast.unparse(method.body[0]).startswith('t = time.time()')
    assert ast.unparse(method.body[1]).startswith('np.random.seed(')
    method.body=method.body[2:]
    names=({'simpleCrop','sample','sampleone','softmax','augment'} if role=='classifier'
           else {'Crop','augment','LabelMapping','select_samples'})
    module=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
                            and n.name in names]+[method],type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
            expression=ast.unparse(node)
            if ((isinstance(node.right,ast.Attribute) and node.right.attr=='stride')
                or (role=='classifier' and isinstance(node.left,ast.Name) and node.left.id=='crop_size')
                or (role=='detector' and (
                    isinstance(node.left,ast.Subscript) and isinstance(node.left.value,ast.Name) and node.left.value.id=='input_size'
                    or 'target' not in expression and isinstance(node.right,ast.Constant) and node.right.value==2
                       and any(name in expression for name in ['crop_size','imgs.shape','bound_size'])))):
                node.op=ast.FloorDiv()
    proxy=SimpleNamespace(**{key:getattr(np,key) for key in dir(np) if not key.startswith('_')})
    proxy.random=rng;proxy.load=lambda _:volume.copy()
    scope=dict(np=proxy,torch=torch,random=label_rng,zoom=zoom,rotate=rotate,warnings=warnings,
               binary_dilation=binary_dilation,generate_binary_structure=generate_binary_structure,
               range=lambda *args:list(range(*args)))
    exec(compile(module,'<pinned-validation-getitem>','exec'),scope)
    return scope


def validate(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    counts={}
    for role in ['classifier','detector']:
        source=checked_source(source_root,pins,'training/classifier/data_'+role+'.py')
        cases=0
        for seed in [0,7,41]:
            for variant in [0,1,7] if role=='classifier' else [2.,24.,46.]:
                rng=np.random.RandomState(seed);other=np.random.RandomState(seed)
                label_rng=random.Random(seed);other_labels=random.Random(seed)
                volume=(np.arange(48**3).reshape(1,48,48,48)%256).astype(np.float32)
                scope=reference(source,role,rng,label_rng,volume)
                if role=='classifier':
                    proposals=np.zeros((variant,5));proposals[:,0]=np.arange(variant)
                    proposals[:,1:4]=2.;proposals[:,4]=10.;known=np.arange(variant)%2
                    config=dict(crop_size=[16]*3,scaleLim=[.85,1.15],radiusLim=[6,100],jitter_range=.15,
                                augtype=dict(scale=True),stride=4,filling_value=160)
                    subject=SimpleNamespace(candidate_box=[proposals],pbb_label=[known],T=1,topk=5,
                        filenames=['synthetic'],random_sample=True,phase='val',crop_size=[16]*3,stride=4,
                        crop=scope['simpleCrop'](config,'val'),yset=[1])
                    actual=classifier_validation_sample(volume,proposals,known,1,topk=5,crop_size=16)
                else:
                    target=np.array([variant]*3+[10.]);boxes=target[None]
                    config=dict(crop_size=[32]*3,bound_size=4,stride=4,pad_value=170,num_neg=800,
                        th_neg=.02,anchors=[10.,30.,60.],th_pos_train=.5,th_pos_val=1.)
                    subject=SimpleNamespace(phase='val',bboxes=np.column_stack(([0],boxes)),
                        filenames=['synthetic'],sample_bboxes=[boxes],augtype=dict(scale=True),
                        crop=scope['Crop'](config),label_mapping=scope['LabelMapping'](config,'val'))
                    actual=detector_validation_sample(volume,target,boxes,crop_size=32,bound_size=4,
                                                      rng=other,label_rng=other_labels)
                expected=scope['__getitem__'](subject,0)
                for a,b in zip(actual,expected):
                    b=b.numpy() if isinstance(b,torch.Tensor) else b
                    np.testing.assert_array_equal(a,b)
                    assert a.dtype==b.dtype,(role,str(a.dtype),str(b.dtype))
                np.testing.assert_array_equal(rng.rand(10),other.rand(10))
                assert label_rng.getstate()==other_labels.getstate()
                cases+=1
        counts[role]=cases
    paths=['sciona/dsb_intake.py','sciona/dsb_components.py','sciona/dsb_detector_crop.py',
           'sciona/dsb_anchor_mapping.py','scripts/validate_dsb_validation_samples.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],exact_sample_cases=counts,
        exact_dtypes_and_rng_continuation=True,
        adaptations=['Clock reseeding replaced by explicit RNG; array loads use synthetic stubs.',
                     'Python 2 integer dimensions and mutable range ported.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_validation_samples.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
