"""Synthetic RGB pixel comparison with hash-pinned historical geometry functions."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import types

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

from sciona.bengali_font_augmentation import FontTransform,augment_font


def definitions(path, digest, names, namespace):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('historical source hash mismatch')
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):
        raise ValueError('missing historical definitions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-geometry-reference>','exec'),namespace)


def main(root,output):
    sources=json.loads(Path('docs/reviews/competition_bengali_affine_source.json').read_text())
    raw=(root/'skimage-0.16.2__geometric.py').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sources['skimage_sources'][0]['sha256']:
        raise ValueError('historical affine source hash mismatch')
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='AffineTransform')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    namespace=dict(np=np,math=math)
    exec(compile(ast.Module(body=[init],type_ignores=[]),'<pinned-affine-init>','exec'),namespace)
    Reference=type('Reference',(),{'__init__':namespace['__init__']})
    def affine(**kwargs):return SimilarityTransform(matrix=Reference(**kwargs).params)
    def gate(array,**kwargs):
        if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[2]!=3:
            raise ValueError('reference adapter restricted to uint8 RGB')
    namespace=dict(np=np,cv2=cv2,
        tf=types.SimpleNamespace(SimilarityTransform=SimilarityTransform,AffineTransform=affine),
        ia=types.SimpleNamespace(is_single_float=lambda value:isinstance(value,float)),
        iadt=types.SimpleNamespace(gate_dtypes=gate),
        _AFFINE_MODE_SKIMAGE_TO_CV2={'constant':cv2.BORDER_CONSTANT},
        _AFFINE_INTERPOLATION_ORDER_SKIMAGE_TO_CV2={1:cv2.INTER_LINEAR})
    definitions(root/'imgaug-0.3.0__geometric.py',sources['imgaug_sources'][1]['source_sha256'],
        ['_AffineSamplingResult','_is_identity_matrix','_warp_affine_arr_cv2'],namespace)
    candidates=json.loads(Path('docs/reviews/competition_bengali_augmentation_candidates.json').read_text())
    source=next(f for f in candidates['artifacts'][0]['files'] if f['module'].endswith('/functional.py'))
    namespace.update(preserve_channel_dim=lambda function:function,get_num_channels=lambda image:image.shape[2])
    definitions(root/'0.4.3__albumentations__augmentations__functional.py',source['sha256'],
        ['pad_with_params','shift_scale_rotate','_maybe_process_in_chunks','random_crop','get_random_crop_coords'],namespace)
    rng=np.random.default_rng(781)
    parameters=[FontTransform(0,0,1,0,0,.5,.5),FontTransform(1e-8,0,1,0,0,0,0),
        FontTransform(5,-5,.9,.0625,-.0625,np.nextafter(1.,0.),0.)]
    parameters.extend(FontTransform(float(rng.uniform(-5,5)),float(rng.uniform(-5,5)),
        float(rng.uniform(.9,1.1)),float(rng.uniform(-.0625,.0625)),float(rng.uniform(-.0625,.0625)),
        float(rng.random()),float(rng.random())) for _ in range(29))
    for p in parameters:
        image=rng.integers(0,256,(224,224,3),dtype=np.uint8)
        expected=namespace['pad_with_params'](image,16,16,16,16,cv2.BORDER_CONSTANT,(255,255,255))
        sample=namespace['_AffineSamplingResult'](scale=(np.ones(1,dtype=np.float32),np.ones(1,dtype=np.float32)),
            translate=([0],[0]),rotate=np.zeros(1,dtype=np.float32),shear=np.array([p.shear],dtype=np.float32))
        matrix,shape=sample.to_matrix(0,expected.shape,False)
        if not namespace['_is_identity_matrix'](matrix):
            expected=namespace['_warp_affine_arr_cv2'](expected,matrix,(255,255,255),'constant',1,shape)
        expected=namespace['shift_scale_rotate'](expected,p.angle,p.scale,p.dx,p.dy,
            cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
        expected=namespace['random_crop'](expected,224,224,p.h_start,p.w_start)
        np.testing.assert_array_equal(augment_font(image,p),expected)
    report=dict(passed=True,synthetic_only=True,catalog_mutations=0,compared_images=len(parameters),
        differing_pixels=0,opencv_version=cv2.__version__,
        sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ['sciona/bengali_font_augmentation.py','scripts/validate_bengali_font_pixels.py']},
        limits=['Explicit parameters only; historical random draws and exact notebook environment remain unqualified.',
                'Both sides use the same OpenCV runtime and installed translation composition.',
                'Reference adapters restrict dtype/channel helpers to the tested uint8 RGB boundary.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.reference_directory,args.output)
