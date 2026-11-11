"""Qualify generic bounded prediction on analytic synthetic applications."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import tf_keras as keras
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sciona.atoms.ml import keras_bounded_prediction as provider

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    cases=[]
    for application in ('synthetic_planar_motion','synthetic_resistive_power'):
        width=2 if application=='synthetic_planar_motion' else 1
        first=keras.Input((width,));second=keras.Input((width,))
        output=(first+.5*second if width==2 else first*first*second)
        model=keras.Model([first,second],output)
        for dtype in (np.float32,np.float64):
            for size in (1,7,33):
                x=np.arange(size*width,dtype=dtype).reshape(size,width)/8
                y=np.full((size,width),2.,dtype=dtype)
                expected=(x+.5*y if width==2 else x*x*y).astype(np.float32)
                for batch in (1,4,32,64):
                    actual=provider.predict_bounded(model,[x,y],batch)
                    assert actual.dtype==np.float32 and actual.shape==expected.shape
                    np.testing.assert_array_equal(actual,expected)
                    cases.append(dict(application=application,input_dtype=str(dtype.__name__),rows=size,
                        columns=width,batch_size=batch,exact_analytic_predictions=True))
    # Batch-normalization inference must preserve its stored running statistics.
    value=keras.Input((2,));normalizer=keras.layers.BatchNormalization()
    model=keras.Model(value,normalizer(value))
    weights=[v.copy() for v in model.get_weights()]
    x=np.arange(14,dtype=np.float32).reshape(7,2)
    expected=model(x,training=False).numpy()
    actual=provider.predict_bounded(model,[x],4)
    np.testing.assert_array_equal(actual,expected)
    for before,after in zip(weights,model.get_weights()):np.testing.assert_array_equal(before,after)
    rejected=[]
    with patch.object(model,'predict',side_effect=AssertionError('Invalid input reached model prediction')):
        for label,inputs,batch in [
            ('empty_inputs',[],4),('not_sequence',x,4),('empty_rows',[x[:0]],4),
            ('integer_dtype',[x.astype(np.int64)],4),('nonfinite',[x*np.nan],4),
            ('misaligned',[x,x[:-1]],4),('scalar',[np.array(1.,dtype=np.float32)],4),
            ('zero_batch',[x],0),('negative_batch',[x],-1),('boolean_batch',[x],True),
            ('floating_batch',[x],4.)]:
            try:provider.predict_bounded(model,inputs,batch)
            except ValueError:rejected.append(label)
            else:raise AssertionError('Invalid contract accepted: '+label)
    for label,output in [('multiple_outputs',[x,x]),('wrong_rows',x[:-1]),
                         ('nonfinite_output',x*np.nan),('integer_output',x.astype(np.int64))]:
        with patch.object(model,'predict',return_value=output):
            try:provider.predict_bounded(model,[x],4)
            except ValueError:rejected.append(label)
            else:raise AssertionError('Invalid output accepted: '+label)
    try:provider.predict_bounded(object(),[x],4)
    except ValueError:rejected.append('non_keras_model')
    else:raise AssertionError('Non-Keras model accepted')
    try:provider.witness_predict_bounded()
    except ValueError:rejected.append('abstract_execution')
    else:raise AssertionError('Abstract execution accepted')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        application_contracts={
            'synthetic_planar_motion': 'Two-dimensional position in metres plus velocity in metres/second times a fixed 0.5-second interval; output position in metres.',
            'synthetic_resistive_power': 'Scalar current in amperes squared times resistance in ohms; output electrical power in watts. These unit meanings are caller declarations, not inferred by the provider.'},
        cases=cases,analytic_cases=len(cases),batch_normalization_inference_exact=True,
        stored_weights_unchanged=True,rejected_contracts=rejected,
        runtime=dict(tensorflow=tf.__version__,legacy_keras=keras.__version__,numpy=np.__version__),
        provider_sha256=sha(inspect.getfile(provider)),implementation_sha256=sha(ROOT/'sciona/keras_bounded_prediction.py'),
        validator_sha256=sha(__file__),resource_policy=dict(native_threads=1,dataset_workers=1,sequential=True),
        limitations=['Constructed mechanical-motion and electrical-power examples only; no empirical model effectiveness claim.',
            'Caller-provisioned trusted legacy-Keras models in the installed runtime; no model loading, authenticity or arbitrary-code isolation.',
            'Single finite floating tensor output only; native Keras conversion and model semantics remain in effect.',
            'Caller must bound compute threads and reserve the additional single dataset worker.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/keras_bounded_provider.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,analytic_cases=result['analytic_cases'],rejected=len(result['rejected_contracts']))))
