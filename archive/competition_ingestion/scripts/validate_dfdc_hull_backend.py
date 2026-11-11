"""Native dlib execution with a predictor trained solely on synthetic landmarks."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import tempfile

import dlib
import numpy as np
import skimage.draw
from skimage import measure

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_hull_backend import load_hull_backend
from sciona.dfdc_convex_hull import blackout_convex_hull


def train_synthetic(parts=68):
    rng=np.random.default_rng(963)
    images=[rng.integers(1,256,(112,112,3),dtype=np.uint8) for _ in range(3)]
    theta=np.linspace(0,2*np.pi,parts,endpoint=False)
    points=np.stack([56+30*np.cos(theta),56+28*np.sin(theta)],axis=1).astype(int)
    if parts==68:
        order=list(range(17))+list(range(26,16,-1))
        theta=np.linspace(0,2*np.pi,len(order),endpoint=False)
        points[order]=np.stack([56+30*np.cos(theta),56+28*np.sin(theta)],axis=1).astype(int)
    rectangle=dlib.rectangle(16,16,95,95)
    labels=[[dlib.full_object_detection(rectangle,[dlib.point(int(x),int(y)) for x,y in points])] for _ in images]
    options=dlib.shape_predictor_training_options()
    options.cascade_depth=2;options.tree_depth=2;options.num_trees_per_cascade_level=5
    options.feature_pool_size=30;options.num_test_splits=5;options.oversampling_amount=3
    options.num_threads=1;options.random_seed='synthetic-963';options.be_verbose=False
    return dlib.train_shape_predictor(images,labels,options),images[0],rectangle,points


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    source=args.source_root/'training/datasets/classifier_dataset.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==next(f['sha256'] for f in pins['files'] if f['path']=='training/datasets/classifier_dataset.py')
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='blackout_convex_hull')
    namespace={'np':np,'random':random,'skimage':skimage,'measure':measure}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<source-hull-native-predictor>','exec'),namespace)
    predictor,image,rectangle,points=train_synthetic()
    result=predictor(image,rectangle)
    assert result.num_parts==68
    coordinates=np.array([[p.x,p.y] for p in result.parts()])
    assert np.max(np.abs(coordinates-points))<=1
    with tempfile.TemporaryDirectory(prefix='sciona-synthetic-dlib-') as directory:
        path=Path(directory)/'predictor.dat';predictor.save(str(path))
        detector,restored=load_hull_backend(predictor_path=path)
        restored_points=np.array([[p.x,p.y] for p in restored(image,rectangle).parts()])
        np.testing.assert_array_equal(restored_points,coordinates)
        assert len(detector(np.zeros((112,112,3),dtype=np.uint8)))==0
        initial=random.getstate();comparisons=0
        try:
            # Positive polygon path uses the actual native trained predictor and
            # a controlled rectangle. This is not a positive HOG-detector test.
            rectangle_detector=lambda image:[rectangle]
            namespace.update(detector=rectangle_detector,predictor=restored)
            for seed in range(8):
                a,b=image.copy(),image.copy()
                random.seed(seed);namespace['blackout_convex_hull'](a);state=random.getstate()
                random.seed(seed);blackout_convex_hull(b,rectangle_detector,restored)
                np.testing.assert_array_equal(a,b);assert random.getstate()==state
                assert np.any(b==0) and np.any(b!=0);comparisons+=1
            blank=np.zeros((112,112,3),dtype=np.uint8)
            namespace.update(detector=detector,predictor=restored)
            a,b=blank.copy(),blank.copy()
            namespace['blackout_convex_hull'](a);blackout_convex_hull(b,detector,restored)
            np.testing.assert_array_equal(a,b)
        finally:random.setstate(initial)
        rejected=0
        five,_,_,_=train_synthetic(parts=5)
        wrong=Path(directory)/'wrong.dat';five.save(str(wrong))
        bad=Path(directory)/'invalid.dat';bad.write_bytes(b'synthetic-invalid-native-state')
        for candidate in (wrong,bad,Path(directory)/'absent.dat'):
            try:load_hull_backend(predictor_path=candidate)
            except ValueError as error:
                assert str(candidate) not in str(error);rejected+=1
            else:raise AssertionError('invalid predictor accepted')
    dist=importlib.metadata.distribution('dlib')
    native=next(p for p in dist.files if str(p).endswith('.so'))
    native_hash=hashlib.sha256(dist.locate_file(native).read_bytes()).hexdigest()
    files=['sciona/dfdc_hull_backend.py','sciona/dfdc_convex_hull.py','scripts/validate_dfdc_hull_backend.py',
           'docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-native-hull-backend-validation.v1','result':'passed','source_commit':pins['commit'],
       'dependency':{'dlib':dlib.__version__,'native_extension_sha256':native_hash},
       'checks':{'actual_synthetic_predictor_training':True,'predicted_landmark_count':68,
                 'native_state_restoration_exact':True,'builtin_frontal_detector_blank_case':True,
                 'native_predictor_source_hull_cases':comparisons,'invalid_predictor_rejections':rejected},
       'state_policy':'Caller explicitly supplies native predictor file; constructor checks native evaluation and68part cardinality. No default predictor path or Python pickle. Synthetic trained states discarded after tests.',
       'limits':'Dlib20.0.1 is an explicit dependency pin; original source dlib version unspecified. Bundled frontal detector exercised on a blank negative fixture. Positive hull tests use actual synthetic-trained predictor with a controlled rectangle, not positive pretrained face detection. No competition images or external pretrained predictor files fetched; no quality/full lifecycle/promotion claim.',
       'documentation':'https://dlib.net/python/index.html#dlib.train_shape_predictor',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_hull_backend.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
