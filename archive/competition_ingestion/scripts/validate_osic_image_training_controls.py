"""Replay pinned callback policy on synthetic scalar models, not B5 training."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PINS = {11: '2cf4af14c6713a73fe4a14a7bb0eb339539a58f531966998753d2425ce277567',
        22: '9c5efe8acb304c3c235753603ba22c1ef8bf988178e3eb39b028baa820477cbe'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(source_directory):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import tf_keras as keras
    import numpy as np
    from sciona import osic_image_training_controls as controls
    from scripts.review_osic_original_scope import TRAIN, REPOSITORY, REVISION

    notebook = json.loads((Path(source_directory)/TRAIN).read_text())
    code = {}
    for index, digest in PINS.items():
        tree = ast.parse(''.join(notebook['cells'][index]['source']))
        assert hashlib.sha256(ast.unparse(tree).encode()).hexdigest() == digest
        code[index] = compile(tree, '<pinned-source-training-policy>', 'exec')
    def model():
        return keras.Sequential([keras.layers.Input((1,)),
                                 keras.layers.Dense(1, use_bias=False, kernel_initializer='ones')])
    oracle = model()
    recorded = {}
    class Recorder:
        def compile(self, **kwargs):
            recorded['compile'] = kwargs
            oracle.compile(**kwargs)
        def fit_generator(self, generator, **kwargs):
            recorded['training_generator'] = generator
            recorded['fit'] = kwargs
            return SimpleNamespace(history={})
    scope = dict(np=np, P=np.arange(3), A=np.zeros(3), TAB=np.zeros((3,4)),
        tf=SimpleNamespace(keras=SimpleNamespace(callbacks=keras.callbacks,
            optimizers=SimpleNamespace(Adam=keras.optimizers.legacy.Adam))),
        build_model=lambda **kwargs: Recorder(), IGenerator=lambda **kwargs: kwargs)
    with redirect_stdout(io.StringIO()):
        exec(code[11], scope)
        exec(code[22], scope)
    assert (scope['EPOCHS'], scope['BATCH_SIZE'], scope['MODEL_CLASS']) == (30,8,'b5')
    fit = recorded['fit']
    assert fit['steps_per_epoch'] == controls.TRAINING_STEPS == 32
    assert fit['validation_steps'] == controls.VALIDATION_STEPS == 16
    assert fit['epochs'] == controls.EPOCHS == 30
    assert scope['BATCH_SIZE'] == controls.BATCH_SIZE
    assert fit['callbacks'] == [scope['cpt'], scope['rlp']]
    assert scope['er'] not in fit['callbacks']
    assert recorded['training_generator']['keys'] is fit['validation_data']['keys']
    candidate = model()
    controls.compile_model(candidate)
    assert oracle.loss == candidate.loss == 'mae'
    assert oracle.optimizer.get_config() == candidate.optimizer.get_config()
    losses = [10.,9.,8.99995] + [9.]*12 + [8.]*15
    assert len(losses) == 30
    expected_rates = []
    rate = np.float32(.003)
    for epoch in range(30):
        if epoch in (6,11,20,25):
            rate = np.float32(rate * .5)
        expected_rates.append(float(rate))
    with tempfile.TemporaryDirectory(prefix='sciona-osic-image-policy-') as tmp:
        paths = [Path(tmp)/'source.h5', Path(tmp)/'candidate.h5']
        source_callbacks = fit['callbacks']
        source_callbacks[0].filepath = str(paths[0])
        source_callbacks[0].verbose = source_callbacks[1].verbose = 0
        groups = [source_callbacks, controls.callbacks(paths[1])]
        traces = []
        for current, group in zip((oracle, candidate), groups):
            assert [type(c).__name__ for c in group] == ['ModelCheckpoint', 'ReduceLROnPlateau']
            for callback in group:
                callback.set_model(current)
                callback.set_params({'epochs':30,'steps':32,'verbose':0})
                callback.on_train_begin({})
            trace = []
            for epoch, loss in enumerate(losses):
                current.set_weights([np.array([[epoch+1.]], dtype=np.float32)])
                logs = {'loss':loss+.5,'val_loss':loss}
                for callback in group:
                    callback.on_epoch_end(epoch, logs)
                if epoch == 2:
                    intermediate = keras.models.load_model(group[0].filepath)
                    np.testing.assert_array_equal(intermediate.get_weights()[0], [[3.]])
                trace.append(float(keras.backend.get_value(current.optimizer.learning_rate)))
                assert not current.stop_training
            for callback in group:
                callback.on_train_end({})
            traces.append(trace)
        np.testing.assert_array_equal(traces[0], traces[1])
        np.testing.assert_array_equal(traces[1], expected_rates)
        for path in paths:
            restored = keras.models.load_model(path)
            np.testing.assert_array_equal(restored.get_weights()[0], [[16.]])
            assert int(restored.optimizer.iterations) == 0
            assert float(keras.backend.get_value(restored.optimizer.learning_rate)) == expected_rates[15]
        assert float(candidate.get_weights()[0][0,0]) == 30.
    try:
        controls.callbacks('unsupported.keras')
    except ValueError:
        pass
    else:
        raise AssertionError('Unqualified checkpoint format accepted')
    return dict(format='osic-image-training-controls.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True,
        source=dict(repository=REPOSITORY, revision=REVISION, normalized_cell_sha256=PINS),
        runtime=dict(tensorflow=tf.__version__, legacy_keras=keras.__version__, native_threads=1,
            optimizer_mapping='Historical tf.keras Adam explicitly mapped to tf_keras.optimizers.legacy.Adam'),
        schedule=dict(epochs=30, batch_size=8, training_steps=32, validation_steps=16,
            same_population_for_training_and_validation=True, early_stopping_active=False),
        replay=dict(callback_epochs_per_model=30, models=2, learning_rates=traces[1],
            lr_reduction_epochs_one_based=[7,12,21,26], best_checkpoint_epoch_one_based=16,
            final_model_epoch_one_based=30, checkpoint_reload_passed=True,
            tiny_improvement_below_scheduler_min_delta_preserved=True),
        controls_sha256=sha(ROOT/'sciona/osic_image_training_controls.py'), validator_sha256=sha(__file__),
        limitations=[
            'Synthetic scalar-model callback replay only; no image training or gradient qualification.',
            'Source model/generator construction and fit are intercepted to inspect policy; population sampling and batch aggregation remain unqualified.',
            'Checkpointing executes on scalar models; custom EfficientNet model restoration remains unqualified.',
            'Historical TensorFlow numerical/random equivalence is not claimed.',
            'Same-population validation is not held-out generalization evidence.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    report = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_image_training_controls.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], schedule=report['schedule'], replay=report['replay'])))
