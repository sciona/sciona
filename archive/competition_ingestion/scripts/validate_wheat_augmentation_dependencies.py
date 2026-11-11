"""Qualify scoped aliases and execute every source augmentation operation."""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from sciona.wheat_augmentation_compat import historical_augmentation_api


def main(root,output):
    manifests={}
    for package,artifact_digest in [('albumentations','77bda58c5cfa5ec8dccd360bf75be52449d6bd7a25dce087192e3c78dd2a439a'),
                                    ('imgaug','82a5052d2daabc1b8de04b5fade44d59f3376c429880dd78e759028c67f3c8f3')]:
        manifest=json.loads((root/(package+'_manifest.json')).read_text())
        if manifest['artifact_sha256']!=artifact_digest:raise ValueError('publisher artifact differs')
        for name,digest in manifest['files_sha256'].items():
            if hashlib.sha256((root/package/name).read_bytes()).hexdigest()!=digest:raise ValueError('historical package source drift')
        manifests[package]=manifest
    raw=(root/'numpy118_type_aliases.py').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='140d8fcf938aa9c2e5d5028b24dbbe1131c8cd0d628b72c7e6949d1e16647d90':
        raise ValueError('historical NumPy type enumeration drift')
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in {'_add_array_type','_set_array_types'}]
    namespace=dict(allTypes=np.sctypeDict,dtype=np.dtype,sctypes={k:[] for k in ('int','uint','float','complex')})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<historical-type-enumeration>','exec'),namespace)
    namespace['_set_array_types']()
    before_types=np.__dict__.get('sctypes');before_iterable=collections.__dict__.get('Iterable')
    rows=[]
    with historical_augmentation_api():
        assert np.sctypes['float']==namespace['sctypes']['float']
        import albumentations as A
        import imgaug
        assert Path(A.__file__).resolve().is_relative_to((root/'albumentations').resolve())
        assert Path(imgaug.__file__).resolve().is_relative_to((root/'imgaug').resolve())
        assert imgaug.parameters.NP_FLOAT_TYPES==set(np.sctypes['float'])
        names=['HorizontalFlip','VerticalFlip','ToGray','IAAAdditiveGaussianNoise','GaussNoise','MotionBlur',
               'MedianBlur','Blur','CLAHE','IAASharpen','IAAEmboss','RandomBrightnessContrast','HueSaturationValue','Resize']
        image=np.random.default_rng(1437).integers(0,256,(96,96,3),dtype=np.uint8)
        for name in names:
            options=dict(height=64,width=64) if name=='Resize' else {}
            for seed in range(8):
                outputs=[]
                for repeat in range(2):
                    random.seed(seed);np.random.seed(seed);imgaug.seed(seed)
                    transform=getattr(A,name)(p=1,**options)
                    result=transform(image=image.copy())['image']
                    assert result.dtype==np.uint8 and result.shape==((64,64,3) if name=='Resize' else image.shape)
                    outputs.append(result)
                np.testing.assert_array_equal(*outputs)
            rows.append(dict(transform=name,seeded_repeat_cases=8,executed=True))
    assert np.__dict__.get('sctypes') is before_types and collections.__dict__.get('Iterable') is before_iterable
    try:
        with historical_augmentation_api():
            raise RuntimeError('synthetic cleanup probe')
    except RuntimeError:
        pass
    assert np.__dict__.get('sctypes') is before_types and collections.__dict__.get('Iterable') is before_iterable
    files=['sciona/wheat_augmentation_compat.py','scripts/validate_wheat_augmentation_dependencies.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,publisher_packages=manifests,
        source_float_type_enumeration_matches=True,namespace_restored_on_success_and_exception=True,
        transforms=rows,seeded_repeat_cases=112,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Execution and seeded repeatability on installed NumPy/OpenCV/SciPy; historical native numerical equivalence is not claimed.',
                'Individual operations forced on; complete probabilistic policy, box transformations and training remain pending.',
                'Scoped aliases require a dedicated serial augmentation process.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,transforms=len(rows),seeded_repeat_cases=112)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reference-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.reference_root,args.output)
