"""Connect encoded synthetic images, source preparation and actual B5 inference."""
import argparse
import ast
import base64
import hashlib
import importlib.metadata as metadata
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(source_directory,overlay):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import tf_keras as keras
    import numpy as np
    import pandas as pd
    import pydicom
    import cv2
    from pydicom.dataset import Dataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,SecondaryCaptureImageStorage,generate_uid
    from sciona.osic_image_population import prepare_training_population,predict_from_encoded
    from sciona.osic_image_batches import sample_batch
    from sciona.osic_image_model import build_model
    from sciona.osic_image_predictor import make_slice_predictor
    from scripts.review_osic_original_scope import TRAIN,INFERENCE
    from scripts.validate_osic_image_metadata import PINS as META_PINS
    from scripts.validate_osic_image_inference import PINS as INFERENCE_PINS
    from scripts.validate_osic_image_batches import GENERATOR_PIN
    cv2.setNumThreads(0)
    assert cv2.getNumThreads()==1
    for module in (pydicom,cv2,cv2._native):
        assert Path(module.__file__).resolve().is_relative_to(Path(overlay).resolve())
    def source(filename,index,digest):
        notebook=json.loads((Path(source_directory)/filename).read_text())
        tree=ast.parse(''.join(notebook['cells'][index]['source']))
        assert hashlib.sha256(ast.unparse(tree).encode()).hexdigest()==digest
        return compile(tree,'<pinned-synthetic-population>','exec')
    def encoded_image(offset):
        ds=Dataset();ds.file_meta=FileMetaDataset()
        ds.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
        ds.SOPClassUID=SecondaryCaptureImageStorage;ds.SOPInstanceUID=generate_uid()
        ds.Rows=13;ds.Columns=19;ds.SamplesPerPixel=1
        ds.PhotometricInterpretation='MONOCHROME2'
        ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=1
        ds.RescaleIntercept='-17.25';ds.RescaleSlope='1.5'
        ds.PixelData=(np.arange(247).reshape(13,19)+offset-123).astype('<i2').tobytes()
        stream=BytesIO();pydicom.dcmwrite(stream,ds,enforce_file_format=True)
        return stream.getvalue()
    names=['-1.dcm','0.dcm','1.dcm','2.dcm','4.dcm','9.dcm']
    images={entity:[(name,encoded_image(g*71+i*11)) for i,name in enumerate(names)]
            for g,entity in enumerate(['a','b'])}
    images['excluded']=[('ignored.dcm',b'invalid but excluded')]
    table=pd.DataFrame(dict(entity=['b','excluded','a','b','a','b'],time=[3.,0.,0.,-1.,4.,1.],
        target=[8.,5.,-1.,0.,11.,4.],numeric=[47.,31.,53.,99.,18.,100.],
        category_a=['Male','male','female','unknown','Male','female'],
        category_b=['other','Never smoked','Ex-smoker','Currently smokes','other','other']))
    source_table=table.rename(columns=dict(entity='Patient',time='Weeks',target='FVC',numeric='Age',
                                         category_a='Sex',category_b='SmokingStatus'))
    raw={(entity,name):payload for entity,records in images.items() for name,payload in records}
    source_rng=np.random.RandomState(42)
    class SourceNumpy:
        random=source_rng
        linalg=SimpleNamespace(lstsq=lambda a,b:np.linalg.lstsq(a,b,rcond=-1))
        def __getattr__(self,name):
            return getattr(np,name)
    storage=SimpleNamespace(dcmread=lambda path:pydicom.dcmread(BytesIO(raw[tuple(path.split('/')[-2:])])))
    scope=dict(np=SourceNumpy(),pd=pd,cv2=cv2,pydicom=storage,train=source_table,
        tqdm=lambda it:it,Sequence=keras.utils.Sequence,BATCH_SIZE=8,
        os=SimpleNamespace(listdir=lambda path:[name for name,_ in images[path.rstrip('/').split('/')[-1]]]))
    exec(source(TRAIN,15,META_PINS[TRAIN,15]),scope)
    exec(source(TRAIN,16,META_PINS[TRAIN,16]),scope)
    exec(source(TRAIN,17,'603e73e60d97715bf81a1e43c5b77274f48d76c5545c402c73cfa9a4535cb287'),scope)
    exec(source(TRAIN,19,GENERATOR_PIN),scope)
    scope['IGenerator'].BAD_ID=['excluded']
    generator=scope['IGenerator'](scope['P'],scope['A'],scope['TAB'])
    actual=prepare_training_population(table,images,excluded_entities=['excluded'],
        time_coordinate='synthetic_time',response_coordinate='synthetic_response')
    assert actual['entities']==generator.keys==['b','a']
    candidate_rng=np.random.RandomState(42)
    for batch in range(3):
        expected=generator[batch]
        got=sample_batch(actual['images'],actual['metadata'],actual['slopes'],candidate_rng)
        for left,right in zip((*got[0],got[1]),(*expected[0],expected[1])):
            assert left.dtype==right.dtype
            np.testing.assert_array_equal(left,right)
    np.testing.assert_array_equal(candidate_rng.randint(0,1000,size=20),source_rng.randint(0,1000,size=20))
    # Query routing uses the case-sensitive inference metadata policy and must
    # not decode the files that the source's ordinal filter discards.
    baseline=table.drop_duplicates('entity').query("entity != 'excluded'").copy()
    baseline['uncertainty']=[.1,3.]
    query=pd.DataFrame(dict(key=['a_5','b_-2','a_0','b_3'],entity=['a','b','a','b'],
                           time=np.array([5,-2,0,3],dtype=np.int64)))
    inference_images={key:[(name,b'invalid but unselected' if i in (0,5) else payload)
                          for i,(name,payload) in enumerate(images[key])] for key in ('a','b')}
    raw={(entity,name):payload for entity,records in inference_images.items() for name,payload in records}
    baseline_source=baseline.rename(columns=dict(entity='Patient',time='Weeks',target='FVC',numeric='Age',
                                                category_a='Sex',category_b='SmokingStatus',uncertainty='Percent'))
    query_source=pd.DataFrame(dict(Patient_Week=query['key'],FVC=np.zeros(4),Confidence=np.zeros(4)))
    keras.utils.set_random_seed(617)
    model=build_model()
    predictor=make_slice_predictor(model)
    scope=dict(np=np,cv2=cv2,pydicom=storage,models=[SimpleNamespace(predict=predictor)],
        pd=SimpleNamespace(read_csv=lambda path:query_source.copy() if 'sample_submission' in path else baseline_source.copy()),
        os=SimpleNamespace(listdir=lambda path:[name for name,_ in inference_images[path.rstrip('/').split('/')[-1]]]))
    for cell in (6,9,15,17):
        exec(source(INFERENCE,cell,INFERENCE_PINS[cell]),scope)
    result=predict_from_encoded(baseline,query,inference_images,predictor)
    assert result['keys']==scope['sub']['Patient_Week'].tolist()
    np.testing.assert_array_equal(result['values'],scope['sub'][['FVC','Confidence']].to_numpy())
    return dict(source_training_batches=3,source_training_samples=24,training_arrays_and_rng_exact=True,
        source_entity_order_preserved=True,excluded_corrupt_payload_not_decoded=True,
        actual_b5_inference_queries=4,source_inference_exact=True,unselected_corrupt_payloads_not_decoded=4,
        runtime=dict(tensorflow=tf.__version__,legacy_keras=keras.__version__,pydicom=pydicom.__version__,
                     numpy=np.__version__,opencv=cv2.__version__),
        resource_policy=dict(native_threads=1,dataset_workers=1))


def validate(dependencies,source_directory):
    from scripts.validate_osic_image_model import PINS
    from scripts.validate_osic_dicom import WHEEL,WHEEL_SHA
    from scripts.audit_osic_opencv_isolation import audit_distribution,digest
    pins=dict(PINS);pins[WHEEL]=WHEEL_SHA
    dependencies=Path(dependencies)
    for name,value in pins.items():
        if sha(dependencies/name)!=value:
            raise ValueError('Pinned dependency changed')
    before=audit_distribution('opencv-python-headless')
    if before['missing'] or before['mismatched']:
        raise ValueError('OpenCV namespace cannot be isolated')
    distribution=metadata.distribution('opencv-python-headless')
    with tempfile.TemporaryDirectory(prefix='sciona-osic-population-') as temporary:
        overlay=Path(temporary)
        for name in pins:
            if name.endswith('.whl'):
                with zipfile.ZipFile(dependencies/name) as archive:
                    archive.extractall(overlay)
        copied={}
        for entry in distribution.files or []:
            relative=Path(str(entry))
            if not entry.hash or relative.parts[0]!='cv2':
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe package path')
            target=overlay/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(distribution.locate_file(entry),target)
            assert base64.urlsafe_b64encode(digest(target,entry.hash.mode)).rstrip(b'=').decode()==entry.hash.value
            copied[str(relative)]=sha(target)
        env=dict(os.environ,PYTHONPATH=str(overlay)+os.pathsep+str(ROOT),PYTHONDONTWRITEBYTECODE='1',
            TF_USE_LEGACY_KERAS='1',TF_CPP_MIN_LOG_LEVEL='3',MPLCONFIGDIR='/private/tmp/sciona-mpl-cache')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
            env[key]='1'
        process=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
            '--dependencies',str(overlay),'--source-directory',str(source_directory)],
            env=env,capture_output=True,text=True)
        if process.returncode:
            raise RuntimeError('Synthetic population qualification failed:\n'+process.stderr)
        result=json.loads(process.stdout)
        assert copied=={name:sha(overlay/name) for name in copied}
    assert audit_distribution('opencv-python-headless')==before
    files=['sciona/osic_image_population.py','sciona/osic_image_inference.py','sciona/dicom_planes.py',
        'sciona/osic_dicom.py','sciona/osic_image_metadata.py','sciona/affine_trajectory_fit.py',
        'sciona/osic_image_batches.py','sciona/group_member_sampling.py','sciona/osic_image_preprocessing.py',
        'sciona/osic_image_model.py','sciona/osic_image_predictor.py','sciona/keras_bounded_prediction.py',
        'sciona/osic_predictions.py','scripts/validate_osic_image_population.py']
    return dict(format='osic-image-population.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,shared_environment_mutations=0,execution=result,dependency_archive_sha256=pins,
        software_sha256={name:sha(ROOT/name) for name in files},
        limitations=['Actual B5 inference uses seeded random weights; complete training and trained-checkpoint composition remain pending.',
            'The source exclusion values are replaced with an explicit synthetic policy; source filtering code and order are preserved.',
            'Training requires valid eligible images and does not reproduce silently shortened batches after decoder errors.',
            'The historical least-squares default is reconstructed with explicit rcond=-1 in the current runtime.',
            'This does not complete the quantile/image blend, portable trained state, graph publication or empirical validation.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies',required=True)
    parser.add_argument('--source-directory',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.source_directory,args.dependencies)))
    else:
        report=validate(args.dependencies,args.source_directory)
        (ROOT/'docs/reviews/osic_image_population.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['execution']))
