"""Isolated, synthetic OSIC B5 dependency/topology/forward qualification.

No package installation, notebook output loading, training or catalog mutation.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'efficientnet-1.1.0-py3-none-any.whl': '4f8a56d15e9c44e9a62dae35404079724c4a1d9e9bfbcfd2a18077e5eda58bf7',
    'Keras_Applications-1.0.8-py3-none-any.whl': 'df4323692b8c1174af821bf906f1e442e63fa7589bf0f1230a0b6bdc5a810c95',
    'keras-applications-source.zip': 'ff87b242e5de063242132c7c8a1155679c084123b857d25adec626d1374a54b4',
}
REVISION = '3b180cb10eda683dda7913ecee2e6487288d292d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(source_directory, overlay):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import numpy as np
    import efficientnet
    import efficientnet.tfkeras as efn
    import keras_applications
    import tf_keras
    from tensorflow import keras
    from sciona.osic_image_model import build_model
    from scripts.review_osic_original_scope import TRAIN, PINS as SOURCE_PINS, REPOSITORY, REVISION as SOURCE_REVISION

    for package in (efficientnet, keras_applications):
        if not Path(package.__file__).resolve().is_relative_to(Path(overlay).resolve()):
            raise ValueError('Dependency escaped isolated overlay')
    notebook = json.loads((Path(source_directory)/TRAIN).read_text())
    definitions = {n.name: n for n in ast.parse(''.join(notebook['cells'][20]['source'])).body
                   if isinstance(n, ast.FunctionDef)}
    for name in ('get_efficientnet', 'build_model'):
        digest = hashlib.sha256(ast.unparse(definitions[name]).encode()).hexdigest()
        if digest != SOURCE_PINS[name]:
            raise ValueError('Pinned source model changed')

    # Inspect the source dispatch without allocating seven unused networks.
    # This is an explicit oracle substitution, not historical RNG equivalence.
    calls = []
    class Factory:
        def __getattr__(self, name):
            def construct(**kwargs):
                calls.append((name, kwargs))
                return name
            return construct
    scope = {'efn': Factory()}
    exec(compile(ast.Module(body=[definitions['get_efficientnet']], type_ignores=[]),
                 '<pinned-source-dispatch>', 'exec'), scope)
    assert scope['get_efficientnet']('b5', (512, 512, 1)) == 'EfficientNetB5'
    assert calls == [('EfficientNetB'+str(i), dict(input_shape=(512,512,1),
                     weights=None, include_top=False)) for i in range(8)]

    keras.utils.set_random_seed(517)
    candidate = build_model()
    scope = {'tf': tf, 'Model': keras.Model,
             'get_efficientnet': lambda model, shape: efn.EfficientNetB5(
                 input_shape=shape, weights=None, include_top=False)}
    for name in ('Input', 'GlobalAveragePooling2D', 'Concatenate', 'Dropout', 'Dense'):
        scope[name] = getattr(keras.layers, name)
    exec(compile(ast.Module(body=[definitions['build_model']], type_ignores=[]),
                 '<pinned-source-head>', 'exec'), scope)
    oracle = scope['build_model'](model_class='b5')
    oracle.set_weights(candidate.get_weights())
    def compare_layers(left, right):
        assert (type(left).__module__, type(left).__qualname__) == (type(right).__module__, type(right).__qualname__)
        if isinstance(left, keras.Model):
            assert len(left.layers) == len(right.layers)
            for a, b in zip(left.layers, right.layers):
                compare_layers(a, b)
        else:
            a, b = left.get_config(), right.get_config()
            a.pop('name', None)
            b.pop('name', None)
            assert a == b
    compare_layers(candidate, oracle)
    assert candidate.count_params() == oracle.count_params() == 28514709
    backbone = next(layer for layer in candidate.layers if isinstance(layer, keras.Model))
    assert not any(type(layer).__name__ in ('Rescaling', 'Normalization') for layer in backbone.layers)
    assert all(layer.trainable for layer in backbone.layers)
    rng = np.random.default_rng(17)
    max_error = 0.
    for image, metadata in [(np.zeros((1,512,512,1), np.float32), np.zeros((1,4),np.float32)),
                            (rng.normal(size=(1,512,512,1)).astype('float32'),
                             np.array([[.2, 1., 0., 1.]], np.float32))]:
        got = candidate([image, metadata], training=False).numpy()
        expected = oracle([image, metadata], training=False).numpy()
        assert got.shape == (1,1) and np.isfinite(got).all()
        np.testing.assert_array_equal(got, expected)
        max_error = max(max_error, float(np.max(np.abs(got-expected))))
    return dict(parameters=candidate.count_params(), backbone_parameters=backbone.count_params(),
                source_dispatch_constructors=8, executed_backbones_per_model=1,
                synthetic_full_size_cases=2, maximum_absolute_error=max_error,
                tensorflow=tf.__version__, legacy_keras=tf_keras.__version__,
                efficientnet=efficientnet.__version__, keras_applications=keras_applications.__version__,
                source_repository=REPOSITORY, source_revision=SOURCE_REVISION,
                source_function_sha256={name: SOURCE_PINS[name] for name in ('get_efficientnet', 'build_model')},
                recursive_layer_config_parity=True,
                no_internal_input_rescaling=True, all_backbone_layers_trainable=True,
                native_threads=1)


def validate(dependencies, source_directory):
    dependencies = Path(dependencies)
    for name, digest in PINS.items():
        if sha(dependencies/name) != digest:
            raise ValueError('Dependency archive changed: '+name)
    with tempfile.TemporaryDirectory(prefix='sciona-osic-b5-') as tmp:
        overlay = Path(tmp)
        for name in PINS:
            if name.endswith('.whl'):
                with zipfile.ZipFile(dependencies/name) as wheel:
                    wheel.extractall(overlay)
        with zipfile.ZipFile(dependencies/'keras-applications-source.zip') as source:
            files = sorted((overlay/'keras_applications').glob('*.py'))
            assert len(files) == 16
            for path in files:
                assert path.read_bytes() == source.read(
                    'keras-applications-'+REVISION+'/keras_applications/'+path.name)
        env = dict(os.environ, TF_USE_LEGACY_KERAS='1', TF_CPP_MIN_LOG_LEVEL='2',
                   MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',
                   PYTHONPATH=str(overlay)+os.pathsep+str(ROOT))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
            env[key] = '1'
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            '--worker', '--source-directory', str(source_directory), '--dependencies', str(overlay)],
            env=env, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError('Isolated synthetic model check failed:\n'+completed.stderr)
        execution = json.loads(completed.stdout)
    return dict(format='osic-image-model.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, dependency_archive_sha256=PINS,
        keras_applications_revision=REVISION, release_modules_matching_revision=16,
        execution=execution,
        model_sha256=sha(ROOT/'sciona/osic_image_model.py'), validator_sha256=sha(__file__),
        limitations=[
            'Inference topology and forward parity only; image training, gradients, optimizer and callbacks remain unqualified.',
            'The source dispatch eagerly builds eight networks; this adapter builds B5 only. Historical initialization/random-stream equivalence is not claimed.',
            'The pinned source head executes with an explicitly substituted B5-only dispatch and shared weights for numerical comparison.',
            'Historical TensorFlow runtime parity and empirical prediction quality are not established.',
            'Wheel hashes bind these inspected downloads; no claim of signed upstream release authentication.',
            'No approval of the original CDG or external dependency redistribution.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies', required=True)
    parser.add_argument('--source-directory', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.source_directory, args.dependencies)))
    else:
        report = validate(args.dependencies, args.source_directory)
        (ROOT/'docs/reviews/osic_image_model.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report['execution']))
