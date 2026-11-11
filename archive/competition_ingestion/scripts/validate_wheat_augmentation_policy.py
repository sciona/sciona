"""Compare the complete policy with source AST, including box geometry and RNG."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np

from sciona.wheat_augmentation_compat import historical_augmentation_api
from sciona.wheat_augmentation import WheatAugmentation


def main(source,reference_root,output):
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!='8224574f23ae68f559e7f9a5a33bbb9e207dfd5ed76f6145db16dfb38c02c84c':raise ValueError('source policy drift')
    manifests={}
    for package in ('albumentations','imgaug'):
        manifest=json.loads((reference_root/(package+'_manifest.json')).read_text())
        for name,sha in manifest['files_sha256'].items():
            if hashlib.sha256((reference_root/package/name).read_bytes()).hexdigest()!=sha:raise ValueError('package source drift')
        manifests[package]=manifest['artifact_sha256']
    tree=ast.parse(raw)
    get_aug=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_aug')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WheatDataset')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assignments=[n for n in init.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Attribute)
                 and n.targets[0].attr in ('train_transforms','resize_transforms')]
    assert len(assignments)==2
    code=compile(ast.Module(body=[get_aug]+assignments,type_ignores=[]),'<historical-policy>','exec')
    image=np.random.default_rng(1493).integers(0,256,(1024,1024,3),dtype=np.uint8)
    boxes=np.array([[0.,0.,1024.,1024.],[15.5,17.25,197.75,307.5],[511.,400.,721.,899.]])
    cases=0
    with historical_augmentation_api():
        import albumentations as A
        import imgaug
        import cv2
        cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
        assert Path(A.__file__).resolve().is_relative_to((reference_root/'albumentations').resolve())
        assert Path(imgaug.__file__).resolve().is_relative_to((reference_root/'imgaug').resolve())
        for size in (512,640,768,1024):
            for seed in range(32):
                random.seed(seed);np.random.seed(seed);imgaug.seed(seed)
                actual=WheatAugmentation(size)
                a_image,a_boxes=actual.resize(image.copy(),boxes.copy())
                a_image,a_boxes=actual.augment(a_image,a_boxes)
                actual_states=(random.getstate(),np.random.get_state(),imgaug.current_random_state().get_state())
                random.seed(seed);np.random.seed(seed);imgaug.seed(seed)
                owner=SimpleNamespace(img_size=size)
                namespace=dict(vars(A),self=owner)
                exec(code,namespace)
                b=owner.resize_transforms(image=image.copy(),bboxes=boxes.copy(),category_id=np.ones(len(boxes),dtype=int))
                b=owner.train_transforms(**b)
                np.testing.assert_array_equal(a_image,b['image'])
                np.testing.assert_array_equal(a_boxes,np.array(b['bboxes']))
                assert actual_states[0]==random.getstate()
                for a,c in zip(actual_states[1:],(np.random.get_state(),imgaug.current_random_state().get_state())):
                    assert a[0]==c[0] and np.array_equal(a[1],c[1]) and a[2:]==c[2:]
                cases+=1
    files=['sciona/wheat_augmentation.py','sciona/wheat_augmentation_compat.py','scripts/validate_wheat_augmentation_policy.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=digest,
        publisher_artifact_sha256=manifests,exact_policy_cases=cases,detector_sizes=[512,640,768,1024],
        pixels_boxes_and_python_numpy_imgaug_rng_exact=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Policy comparison uses pinned historical Python packages on installed native dependencies; historical native numerical equivalence is not claimed.',
                'Complete dataset sampling/retry loop and full source training remain pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_policy_cases=cases)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--reference-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.reference_root,args.output)
