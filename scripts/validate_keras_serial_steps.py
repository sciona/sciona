"""Check serial step dispatch against native Keras batch methods."""
import hashlib
import json
from pathlib import Path


def validate():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import tf_keras as keras
    import numpy as np
    from sciona.keras_serial_steps import make_steps
    from sciona.osic_image_training_controls import compile_model
    def model():
        result=keras.Sequential([keras.layers.Input((4,)),keras.layers.Dense(3,activation='relu'),
                                 keras.layers.Dense(1)])
        compile_model(result)
        return result
    keras.utils.set_random_seed(19)
    candidate,oracle=model(),model()
    oracle.set_weights(candidate.get_weights())
    train,test=make_steps(candidate)
    rng=np.random.default_rng(17)
    comparisons=0
    for epoch in range(3):
        candidate.reset_metrics();oracle.reset_metrics()
        for size in (8,8,3):
            x=rng.normal(size=(size,4)).astype('float32')
            y=rng.normal(size=size).astype('float32')
            expected=oracle.train_on_batch(x,y,reset_metrics=False,return_dict=True)
            actual=train(x,y)
            np.testing.assert_array_equal(float(actual['loss']),expected['loss'])
            for a,b in zip(candidate.get_weights(),oracle.get_weights()):
                np.testing.assert_array_equal(a,b)
            comparisons+=1
        candidate.reset_metrics();oracle.reset_metrics()
        for size in (8,3):
            x=rng.normal(size=(size,4)).astype('float32')
            y=rng.normal(size=size).astype('float32')
            expected=oracle.test_on_batch(x,y,reset_metrics=False,return_dict=True)
            actual=test(x,y)
            np.testing.assert_array_equal(float(actual['loss']),expected['loss'])
            comparisons+=1
    assert int(candidate.optimizer.iterations)==int(oracle.optimizer.iterations)==9
    root=Path(__file__).resolve().parents[1]
    return dict(format='keras-serial-steps.v1',passed=True,synthetic_only=True,
        approved=False,catalog_mutations=0,training_updates=9,
        exact_loss_comparisons=comparisons,exact_weight_comparisons_after_every_update=True,
        unequal_batch_metric_aggregation=True,tensorflow=tf.__version__,legacy_keras=keras.__version__,
        step_sha256=hashlib.sha256((root/'sciona/keras_serial_steps.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Dense deterministic synthetic model; stochastic B5 training equivalence is not established.',
                     'No dataset-loader scheduling, full workflow or provider approval qualification.'])


if __name__=='__main__':
    result=validate()
    root=Path(__file__).resolve().parents[1]
    (root/'docs/reviews/keras_serial_steps.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
