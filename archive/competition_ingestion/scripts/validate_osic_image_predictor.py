"""Compare real B5 prediction with the native Keras default batch adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import tf_keras as keras
    import numpy as np
    from tf_keras.src.engine.data_adapter import TensorLikeDataAdapter
    from sciona.osic_image_model import build_model
    from sciona.osic_image_predictor import make_slice_predictor
    keras.utils.set_random_seed(617)
    model=build_model()
    def weight_identity():
        digest=hashlib.sha256()
        for value in model.get_weights():
            digest.update(value.tobytes())
        return digest.hexdigest()
    before=weight_identity()
    rng=np.random.default_rng(822)
    inputs=[rng.normal(size=(33,512,512,1)),rng.normal(size=(33,4))]
    adapter=TensorLikeDataAdapter(inputs,shuffle=False)
    assert adapter.batch_size()==32 and adapter.get_size()==2 and adapter.partial_batch_size()==1
    options=tf.data.Options()
    options.threading.private_threadpool_size=1
    options.threading.max_intra_op_parallelism=1
    options.autotune.enabled=False
    dataset=adapter.get_dataset().with_options(options)
    expected=model.predict(dataset,verbose=0)
    actual=make_slice_predictor(model)(inputs)
    assert expected.shape==actual.shape==(33,1)
    assert expected.dtype==actual.dtype==np.float32
    np.testing.assert_array_equal(actual,expected)
    assert weight_identity()==before
    # Reject input misalignment before invoking any model computation.
    predictor=make_slice_predictor(model)
    rejected=0
    for invalid in ([inputs[0],inputs[1][:-1]], [inputs[0][:0],inputs[1][:0]],
                    [np.full((1,512,512,1),np.nan),np.ones((1,4))]):
        try:
            predictor(invalid)
        except ValueError:
            rejected+=1
        else:
            raise AssertionError('Invalid prediction inputs accepted')
    return dict(queries=33,source_default_batch_size=32,source_batches=2,partial_batch_size=1,
        full_size_images=True,exact_source_predictions=True,maximum_absolute_error=0.,
        learned_weights_unchanged=True,rejected_input_contracts=rejected,
        runtime=dict(tensorflow=tf.__version__,legacy_keras=keras.__version__,numpy=np.__version__),
        resource_policy=dict(native_threads=1,reference_dataset_workers=1,candidate_dataset_workers=1,
                             reference_and_candidate_sequential=True))


def validate(dependencies):
    from scripts.validate_osic_image_model import PINS
    dependencies=Path(dependencies)
    for name,digest in PINS.items():
        if sha(dependencies/name)!=digest:
            raise ValueError('Dependency archive changed')
    with tempfile.TemporaryDirectory(prefix='sciona-osic-predictor-') as overlay:
        for name in PINS:
            if name.endswith('.whl'):
                with zipfile.ZipFile(dependencies/name) as wheel:
                    wheel.extractall(overlay)
        env=dict(os.environ,TF_USE_LEGACY_KERAS='1',TF_CPP_MIN_LOG_LEVEL='3',
            MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=overlay+os.pathsep+str(ROOT))
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS',
                    'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
            env[key]='1'
        process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],
            env=env,capture_output=True,text=True)
        if process.returncode:
            raise RuntimeError('Isolated B5 prediction failed:\n'+process.stderr)
        result=json.loads(process.stdout)
    files=['sciona/keras_bounded_prediction.py','sciona/osic_image_predictor.py',
           'sciona/osic_image_model.py','scripts/validate_osic_image_predictor.py']
    return dict(format='osic-image-predictor.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,execution=result,dependency_archive_sha256=PINS,
        software_sha256={name:sha(ROOT/name) for name in files},
        limitations=['Seeded random B5 weights, not the ongoing training checkpoint or empirical effectiveness.',
            'Native Keras array-adapter data order and default batch size retained; its worker pool is explicitly limited to one.',
            'This qualifies batch dispatch and floating input conversion, not raw DICOM decoding or full branch composition.',
            'Historical runtime numerical parity and original CDG approval remain unqualified.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker()))
    else:
        if not args.dependencies:
            parser.error('--dependencies required')
        report=validate(args.dependencies)
        (ROOT/'docs/reviews/osic_image_predictor.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['execution']))
