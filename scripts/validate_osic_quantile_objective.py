"""Compare source loss values/gradients and explicit legacy Adam updates."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
             'TF_NUM_INTEROP_THREADS'):
    os.environ[name] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/sciona-mpl-cache')

# Import TensorFlow first in this dedicated serial worker.
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import tf_keras
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np

from sciona import osic_quantile_objective as adapter
from sciona import tensorflow_distribution_losses as generic
from scripts.review_osic_original_scope import ROOT, INFERENCE, REVISION, PINS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_gradient(fn, targets, predictions):
    variable = tf.Variable(predictions, dtype=tf.float32)
    with tf.GradientTape() as tape:
        value = fn(tf.constant(targets, tf.float32), variable)
    gradient = tape.gradient(value, variable)
    if gradient is None or not np.isfinite(gradient.numpy()).all():
        raise ValueError('Complete finite prediction gradient required')
    return value.numpy(), gradient.numpy()


def validate(directory):
    paths = [Path(__file__), ROOT/'sciona/osic_quantile_objective.py',
        ROOT/'sciona/tensorflow_distribution_losses.py',
        Path(inspect.getfile(tf_keras.optimizers.legacy.Adam))]
    before = {('matcher/'+str(p.relative_to(ROOT))) if p.is_relative_to(ROOT)
              else 'legacy_optimizer_source': sha(p) for p in paths}
    notebook = json.loads((Path(directory)/INFERENCE).read_text())
    definitions = {node.name: node for node in ast.parse(''.join(notebook['cells'][32]['source'])).body
                   if isinstance(node, ast.FunctionDef)}
    namespace = dict(tf=tf, np=np, K=tf_keras.backend,
                     C1=tf.constant(70, tf.float32), C2=tf.constant(1000, tf.float32))
    for name in ('score', 'qloss', 'mloss'):
        node = definitions[name]
        if hashlib.sha256(ast.unparse(node).encode()).hexdigest() != PINS[name]:
            raise ValueError('Pinned source objective changed')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-source-loss>', 'exec'), namespace)
    rng = np.random.default_rng(673)
    cases = [
        ('continuous', rng.normal(100, 150, (7, 1)), rng.normal(100, 150, (7, 3))),
        ('clipping_boundaries', np.array([[0.], [1000.], [1001.], [-1001.]]),
         np.array([[0., 0., 70.], [-35., 0., 35.], [0., 0., 69.], [80., 0., -80.]])),
        ('crossed_outputs', np.array([[2.], [-3.]]), np.array([[4., 2., 0.], [5., -3., -8.]])),
    ]
    checks = []
    for label, target, predictions in cases:
        for name, actual_fn, reference_fn in [('pinball', adapter.quantile_loss, namespace['qloss']),
                ('score', adapter.score, namespace['score']),
                ('mixed', adapter.mixed_loss, namespace['mloss'](.8))]:
            actual, actual_gradient = value_gradient(actual_fn, target, predictions)
            expected, expected_gradient = value_gradient(reference_fn, target, predictions)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_gradient, expected_gradient)
            checks.append(dict(case=label, objective=name, value_exact=True, gradient_exact=True))
    reuse = []
    for label, floor, cap in [('synthetic-duration', 2., 30.), ('synthetic-charge', .01, .8)]:
        target = np.array([[.2], [.5], [.9]], dtype=np.float32)
        prediction = np.array([[.1, .3, .5], [.7, .5, .2], [.1, 1., 1.2]], dtype=np.float32)
        expected = np.mean([min(abs(float(y)-float(p[1])), cap)/max(float(p[2])-float(p[0]), floor)*np.sqrt(2)
            + np.log(max(float(p[2])-float(p[0]), floor)*np.sqrt(2)) for y,p in zip(target[:, 0], prediction)])
        actual = generic.clipped_laplace_score(target, prediction, floor, cap).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)
        reuse.append(dict(case=label, explicit_policy_matches_scalar_reference=True))
    target = np.zeros((2, 1), dtype=np.float32)
    prediction = np.ones((2, 3), dtype=np.float32)
    invalid = [
        ('wrong_dtype', lambda: adapter.score(target.astype(np.float64), prediction)),
        ('wrong_target_rank', lambda: adapter.score(target[:, 0], prediction)),
        ('nonfinite_prediction', lambda: adapter.score(target, prediction*np.nan)),
        ('quantile_column_mismatch', lambda: generic.mean_pinball(target, prediction, [.2, .8])),
        ('invalid_quantile', lambda: generic.mean_pinball(target, prediction, [.2, .5, 1.1])),
        ('zero_scale', lambda: generic.clipped_laplace_score(target, prediction, 0., 1.)),
        ('scale_underflow', lambda: generic.clipped_laplace_score(target, prediction, 1e-100, 1.)),
        ('scale_overflow', lambda: generic.clipped_laplace_score(target, prediction, 1e100, 1.)),
    ]
    rejected = []
    for label, operation in invalid:
        try:
            operation()
        except (ValueError, tf.errors.InvalidArgumentError):
            rejected.append(label)
        else:
            raise ValueError('Invalid loss contract accepted: '+label)
    opt = adapter.optimizer()
    if opt.epsilon != 1e-7:
        raise ValueError('Legacy backend epsilon differs from qualified default')
    variable = tf.Variable(np.array([.3, -.8, 1.2], dtype=np.float32))
    expected = variable.numpy().astype(np.float64)
    first = np.zeros(3); second = np.zeros(3); maximum = 0.
    for step in range(40):
        gradient = rng.normal(size=3).astype(np.float32)
        first = .9 * first + .1 * gradient.astype(np.float64)
        second = .999 * second + .001 * np.square(gradient.astype(np.float64))
        rate = .1/(1+.01*step)
        corrected_rate = rate*np.sqrt(1-.999**(step+1))/(1-.9**(step+1))
        expected -= corrected_rate*first/(np.sqrt(second)+1e-7)
        opt.apply_gradients([(tf.constant(gradient), variable)])
        maximum = max(maximum, float(np.max(np.abs(variable.numpy()-expected))))
        np.testing.assert_allclose(variable.numpy(), expected, rtol=2e-5, atol=3e-6)
    if int(opt.iterations.numpy()) != 40:
        raise ValueError('Optimizer iteration count differs')
    np.testing.assert_allclose(opt._decayed_lr(tf.float32).numpy(), .1/(1+.01*40), rtol=2e-7)
    after = {('matcher/'+str(p.relative_to(ROOT))) if p.is_relative_to(ROOT)
             else 'legacy_optimizer_source': sha(p) for p in paths}
    if before != after:
        raise ValueError('Implementation changed during qualification')
    return dict(format='osic-quantile-objective-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, source_revision=REVISION,
        source_function_sha256={name:PINS[name] for name in ('score', 'qloss', 'mloss')},
        source_comparisons=checks, explicit_policy_cases=reuse, rejected_contract_cases=rejected,
        optimizer_probe=dict(implementation='tf_keras.optimizers.legacy.Adam', updates=40,
            epsilon=opt.epsilon, inverse_time_decay=.01, base_learning_rate=.1,
            maximum_absolute_error=maximum, float64_equation_reference=True),
        software_sha256=before, runtime=dict(tensorflow=tf.__version__, tf_keras=tf_keras.__version__, numpy=np.__version__),
        resource_policy=dict(native_threads=1, tensorflow_intraop_threads=1, tensorflow_interop_threads=1),
        limitations=['Loss values and prediction gradients match pinned source under the current TensorFlow backend.',
            'Legacy Adam dense float32 updates match independent equations within stated tolerances; historical backend bitwise equality is not claimed.',
            'No complete model training, feature preparation, package provenance or original CDG publication is qualified.',
            'Quantile ordering and calibrated coverage are not guaranteed by these objectives.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    report = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_quantile_objective_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, source_gradient_checks=len(report['source_comparisons']),
        optimizer_updates=report['optimizer_probe']['updates'], approved=False)))
