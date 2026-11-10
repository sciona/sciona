"""Run both complete source quantile schedules against a pinned model oracle.

Synthetic prepared features only. Each candidate and source oracle receives
identical initialized weights and explicit batch order. This qualifies the
architecture and updates, not historical random draws or feature preparation.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS',
             'TF_NUM_INTEROP_THREADS'):
    os.environ[name] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/sciona-mpl-cache')
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import tf_keras as keras
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from sciona.osic_quantile_model import build_model
from scripts.review_osic_original_scope import ROOT, INFERENCE, REVISION, PINS


class SourceTensorFlow:
    """Declare the optimizer compatibility substitution rather than hide it."""
    keras = SimpleNamespace(optimizers=SimpleNamespace(Adam=keras.optimizers.legacy.Adam))

    def __getattr__(self, name):
        return getattr(tf, name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_factory(directory):
    notebook = json.loads((Path(directory)/INFERENCE).read_text())
    definitions = {n.name:n for n in ast.parse(''.join(notebook['cells'][32]['source'])).body
                   if isinstance(n, ast.FunctionDef)}
    namespace = dict(tf=SourceTensorFlow(), np=np, K=keras.backend, L=keras.layers, M=keras.models,
                     C1=tf.constant(70, tf.float32), C2=tf.constant(1000, tf.float32))
    for name in ('score', 'qloss', 'mloss', 'make_model'):
        node = definitions[name]
        if hashlib.sha256(ast.unparse(node).encode()).hexdigest() != PINS[name]:
            raise ValueError('Pinned model source changed')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-model-source>', 'exec'), namespace)
    return namespace['make_model']


def step_function(model):
    @tf.function(input_signature=[tf.TensorSpec([None, 7], tf.float32), tf.TensorSpec([None, 1], tf.float32)])
    def step(features, targets):
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = model.loss(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    return step


def validate(directory):
    paths = [Path(__file__), ROOT/'sciona/osic_quantile_model.py', ROOT/'sciona/osic_quantile_objective.py',
             ROOT/'sciona/tensorflow_distribution_losses.py']
    before = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    factory = source_factory(directory)
    rng = np.random.default_rng(190)
    # Constructed feature matrix only: no clinical records or source table schema.
    features = rng.uniform(-1., 1., (32, 7)).astype(np.float32)
    targets = (3*features[:, :1] - 2*features[:, 1:2] + .2).astype(np.float32)
    queries = rng.uniform(-1., 1., (9, 7)).astype(np.float32)
    folds = []
    for fold in range(2):
        keras.backend.clear_session()
        keras.utils.set_random_seed(381+fold)
        reference = factory(7)
        actual = build_model(7)
        actual.set_weights(reference.get_weights())
        if actual.count_params() != reference.count_params() or actual.count_params() != 11506:
            raise ValueError('Complete dense architecture differs')
        np.testing.assert_array_equal(actual(queries).numpy(), reference(queries).numpy())
        heldout = np.arange(fold*16, (fold+1)*16)
        training = np.setdiff1d(np.arange(32), heldout)
        candidate_step, reference_step = step_function(actual), step_function(reference)
        max_loss_error = 0.; max_weight_error = 0.; max_prediction_error = 0.
        for epoch in range(600):
            order = rng.permutation(training)
            # Source batch size is 128; this synthetic fold fits in one batch.
            x, y = features[order], targets[order]
            candidate_loss = candidate_step(x, y).numpy()
            reference_loss = reference_step(x, y).numpy()
            if not np.isfinite(candidate_loss) or not np.isfinite(reference_loss):
                raise ValueError('Finite loss required throughout both schedules')
            max_loss_error = max(max_loss_error, float(abs(candidate_loss-reference_loss)))
            np.testing.assert_allclose(candidate_loss, reference_loss, rtol=2e-6, atol=2e-6)
            for candidate_weight, reference_weight in zip(actual.get_weights(), reference.get_weights()):
                error = float(np.max(np.abs(candidate_weight-reference_weight)))
                max_weight_error = max(max_weight_error, error)
                np.testing.assert_allclose(candidate_weight, reference_weight, rtol=2e-6, atol=2e-6)
            candidate_validation = actual(features[heldout], training=False).numpy()
            reference_validation = reference(features[heldout], training=False).numpy()
            max_prediction_error = max(max_prediction_error, float(np.max(np.abs(candidate_validation-reference_validation))))
            np.testing.assert_allclose(candidate_validation, reference_validation, rtol=2e-6, atol=2e-6)
            if (epoch+1) % 200 == 0:
                print(json.dumps(dict(fold=fold, completed_epochs=epoch+1)), flush=True)
        if int(actual.optimizer.iterations.numpy()) != 600 or int(reference.optimizer.iterations.numpy()) != 600:
            raise ValueError('Full source update schedule required')
        np.testing.assert_allclose(actual(queries).numpy(), reference(queries).numpy(), rtol=2e-6, atol=2e-6)
        restored = build_model(7)
        restored.set_weights(actual.get_weights())
        np.testing.assert_array_equal(restored(queries).numpy(), actual(queries).numpy())
        folds.append(dict(fold=fold, epochs=600, candidate_updates=600, oracle_updates=600,
            training_rows=16, validation_rows=16, validation_checks=600, query_rows=9,
            parameter_count=actual.count_params(), maximum_loss_error=max_loss_error,
            maximum_weight_error=max_weight_error, maximum_validation_prediction_error=max_prediction_error,
            numeric_weight_reload_exact=True))
    if before != {str(p.relative_to(ROOT)):sha(p) for p in paths}:
        raise ValueError('Implementation changed during training')
    return dict(format='osic-quantile-model-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, source_revision=REVISION,
        source_function_sha256={name:PINS[name] for name in ('score','qloss','mloss','make_model')},
        folds=folds, software_sha256=before,
        runtime=dict(tensorflow=tf.__version__, tf_keras=keras.__version__, numpy=np.__version__),
        resource_policy=dict(native_threads=1, tensorflow_intraop_threads=1, tensorflow_interop_threads=1,
                             serial_models=True, data_loader_workers=0),
        compatibility=['Pinned source Adam constructor is explicitly mapped to tf_keras.optimizers.legacy.Adam.',
            'Both models receive identical initial weights and batch permutations; historical initialization and shuffle sequences are not reproduced.',
            'Explicit GradientTape steps avoid a parallel data loader; no regularizers, batch-normalization updates or dropout exist in this source quantile model.'],
        limitations=['Two full 600-epoch schedules on synthetic prepared features only; the source population and full preprocessing are not reproduced.',
            'One batch per synthetic fold exercises the full epoch count but not historical population size or update count.',
            'Numeric weight reload only; optimizer-state persistence and independent-process replay remain unqualified.',
            'The image training branch, complete graph, package provenance and catalog publication remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    report = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_quantile_model_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, folds=len(report['folds']), candidate_updates=1200, approved=False)))
