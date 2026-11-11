"""Pinned classifier training loader sequence using synthetic array IO stubs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from scipy.ndimage import zoom, rotate

from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_training import classifier_training_sample, detector_training_sample
from sciona.dsb_schedule import CLASSIFIER_PROFILES


def validate(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    tree=ast.parse(checked_source(source_root,pins,'training/classifier/data_classifier.py'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataBowl3Classifier')
    getitem=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    start=next(i for i,n in enumerate(getitem.body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='pbb')
    # Replace wall-clock seeding with caller RNG while retaining every subsequent
    # source sampling/cropping/augmentation instruction.
    getitem.body=getitem.body[start:]
    selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
              and n.name in ['simpleCrop','sample','sampleone','softmax','augment']]+[getitem]
    tree=ast.Module(body=selected,type_ignores=[])
    for node in ast.walk(tree):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
            if ((isinstance(node.left,ast.Name) and node.left.id=='crop_size')
                    or (isinstance(node.right,ast.Attribute) and node.right.attr=='stride')):
                node.op=ast.FloorDiv()
    code=compile(tree,'<pinned-training-classifier-sequence>','exec')
    cases=0
    for variant in [3,4]:
        flags={k:CLASSIFIER_PROFILES[variant][k] for k in ['flip','swap','rotate','scale']}
        for count in [0,1,7]:
            for seed in [0,7,41]:
                volume=np.arange(25**3,dtype=np.float32).reshape(1,25,25,25)%256
                proposals=np.zeros((count,5),dtype=float)
                proposals[:,0]=np.arange(count);proposals[:,1:4]=12;proposals[:,4]=10
                known=np.arange(count)%2
                original_rng=np.random.RandomState(seed);candidate_rng=np.random.RandomState(seed)
                proxy=SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')})
                proxy.random=original_rng;proxy.load=lambda _:volume.copy()
                scope=dict(np=proxy,torch=torch,zoom=zoom,rotate=rotate,warnings=warnings,
                           range=lambda *args:list(range(*args)))
                exec(code,scope)
                config=dict(crop_size=[16]*3,scaleLim=[.85,1.15],radiusLim=[6,100],jitter_range=.15,
                            augtype=flags,stride=4,filling_value=160)
                subject=SimpleNamespace(candidate_box=[proposals],pbb_label=[known],T=1,topk=5,
                    filenames=['synthetic'],random_sample=True,phase='train',crop_size=[16]*3,
                    stride=4,augtype=flags,crop=scope['simpleCrop'](config,'train'),yset=[1])
                expected=scope['__getitem__'](subject,0)
                actual=classifier_training_sample(volume,proposals,known,topk=5,crop_size=16,
                                                   rng=candidate_rng,**flags)
                for a,b in zip(actual[:3],expected[:3]):np.testing.assert_array_equal(a,b.numpy())
                np.testing.assert_array_equal(original_rng.rand(10),candidate_rng.rand(10))
                cases+=1
    import random
    from scipy.ndimage import binary_dilation,generate_binary_structure
    detector_tree=ast.parse(checked_source(source_root,pins,'training/classifier/data_detector.py').expandtabs(8))
    cls=next(n for n in detector_tree.body if isinstance(n,ast.ClassDef) and n.name=='DataBowl3Detector')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    statements=[]
    for node in ast.walk(method):
        if isinstance(node,ast.Assign):
            expression=ast.unparse(node)
            if expression.startswith('sample, target, bboxes, coord = self.crop(imgs, bbox[1:]'):
                statements.append(node)
            elif expression.startswith('label = self.label_mapping(') or expression=='sample = sample.astype(np.float32)':
                statements.append(node)
        elif isinstance(node,ast.If) and ast.unparse(node.test)=="self.phase == 'train' and (not isRandom)":
            statements.append(node)
    statements.sort(key=lambda n:n.lineno)
    assert len(statements)==4, [ast.unparse(n) for n in statements]
    sequence=compile(ast.Module(body=statements,type_ignores=[]),'<pinned-selected-detector-sequence>','exec')
    helpers=ast.Module(body=[n for n in detector_tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
                            and n.name in ['Crop','augment','LabelMapping','select_samples']],type_ignores=[])
    for node in ast.walk(helpers):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
            expression=ast.unparse(node)
            if ((isinstance(node.right,ast.Attribute) and node.right.attr=='stride')
                    or (isinstance(node.left,ast.Subscript) and isinstance(node.left.value,ast.Name) and node.left.value.id=='input_size')
                    or ('target' not in expression and isinstance(node.right,ast.Constant) and node.right.value==2
                        and any(name in expression for name in ['crop_size','imgs.shape','bound_size']))):
                node.op=ast.FloorDiv()
    helper_code=compile(helpers,'<pinned-detector-training-helpers>','exec')
    detector_cases=0
    for seed in [0,7,41]:
        for random_crop,missing,scaling in [(False,False,False),(False,False,True),
                                           (True,False,False),(True,False,True),(True,True,False)]:
            original_rng=np.random.RandomState(seed);candidate_rng=np.random.RandomState(seed)
            original_labels=random.Random(seed);candidate_labels=random.Random(seed)
            proxy=SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')})
            proxy.random=original_rng
            scope=dict(np=proxy,random=original_labels,zoom=zoom,rotate=rotate,warnings=warnings,
                       binary_dilation=binary_dilation,generate_binary_structure=generate_binary_structure)
            exec(helper_code,scope)
            config=dict(crop_size=[32]*3,bound_size=4,stride=4,pad_value=170,num_neg=20,
                        th_neg=.02,anchors=[10.,30.,60.],th_pos_train=.5,th_pos_val=1.)
            volume=np.arange(48**3,dtype=np.float32).reshape(1,48,48,48)%256
            target=None if missing else np.array([24.,24.,24.,10.])
            boxes=np.array([[24.,24.,24.,10.],[20.,21.,22.,4.]])
            flags=dict(flip=True,rotate=False,swap=False)
            scope.update(self=SimpleNamespace(crop=scope['Crop'](config),phase='train',augtype=flags,
                         label_mapping=scope['LabelMapping'](config,'train')),
                         imgs=volume,bbox=[] if missing else np.r_[0,target],bboxes=boxes.copy(),
                         isScale=scaling,isRandom=random_crop)
            exec(sequence,scope)
            actual=detector_training_sample(volume,target,boxes,crop_size=32,bound_size=4,num_neg=20,
                scale=scaling,random_crop=random_crop,rng=candidate_rng,label_rng=candidate_labels,**flags)
            for a,b in zip(actual,[scope['sample'],scope['label'],scope['coord']]):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(original_rng.rand(10),candidate_rng.rand(10))
            assert original_labels.getstate()==candidate_labels.getstate()
            detector_cases+=1
    paths=['sciona/dsb_training.py','sciona/dsb_components.py','sciona/dsb_detector_crop.py',
           'sciona/dsb_anchor_mapping.py','scripts/validate_dsb_training_sequence.py','tests/test_dsb_training.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],
                classifier_training_sequence_exact_array_and_rng_cases=cases,
                detector_selected_training_sequence_exact_array_and_both_rng_cases=detector_cases,
                source_adaptations=['Clock seed replaced by explicit shared RNG; file loading returns synthetic arrays.',
                                   'Python 2 mutable range and integer coordinate dimensions ported.'],
                limitations=['Detector sequence starts after image/target selection; dataset-level sampling still requires validation.',
                             'Optimizer epoch parity, validation and checkpoint lifecycle remain.',
                             'Training uses raw detector intensities and zero classifier rotation padding, unlike root loaders.'],
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_training_sequence_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
