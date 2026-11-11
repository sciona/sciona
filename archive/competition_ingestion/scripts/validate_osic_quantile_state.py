"""Verify numeric state in a fresh process with training entrypoints disabled."""
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
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

from sciona.osic_quantile_model import build_model
from sciona.osic_quantile_state import export_prediction_state, restore_prediction_state

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['f'+str(i) for i in range(7)]
FEATURE = 'synthetic-features.v1'
RESPONSE = 'synthetic-response.v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(directory):
    def forbidden(*args, **kwargs):
        raise AssertionError('Training is forbidden during prediction-state replay')
    keras.Model.fit = forbidden
    keras.Model.train_on_batch = forbidden
    keras.optimizers.legacy.Adam.apply_gradients = forbidden
    state = json.loads((directory/'state.json').read_text())
    arrays = np.load(directory/'queries.npz', allow_pickle=False)
    model = restore_prediction_state(state, COLUMNS, FEATURE, RESPONSE)
    np.testing.assert_array_equal(model(arrays['queries'], training=False).numpy(), arrays['expected'])
    if int(model.optimizer.iterations.numpy()) != 0:
        raise ValueError('Replay performed optimizer updates')
    return dict(passed=True, queries=len(arrays['queries']), exact_predictions=True,
                optimizer_updates=0, training_entrypoints_disabled=True)


def validate():
    paths = [Path(__file__), ROOT/'sciona/osic_quantile_state.py', ROOT/'sciona/osic_quantile_model.py',
             ROOT/'sciona/osic_quantile_objective.py', ROOT/'sciona/tensorflow_distribution_losses.py']
    before = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    keras.utils.set_random_seed(456)
    rng = np.random.default_rng(749)
    x = rng.uniform(-1, 1, (24, 7)).astype(np.float32)
    y = (2*x[:, :1] + x[:, 1:2]).astype(np.float32)
    queries = rng.uniform(-1, 1, (13, 7)).astype(np.float32)
    model = build_model(7)
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = model.loss(y, model(x, training=True))
        model.optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
    for _ in range(20):
        train_step()
    state = export_prediction_state(model, COLUMNS, FEATURE, RESPONSE)
    encoded = json.dumps(state, allow_nan=False)
    state = json.loads(encoded)
    expected = model(queries, training=False).numpy()
    np.testing.assert_array_equal(restore_prediction_state(state, COLUMNS, FEATURE, RESPONSE)(queries).numpy(), expected)
    rejected = []
    def reject(name, mutate):
        changed = copy.deepcopy(state)
        mutate(changed)
        try:
            restore_prediction_state(changed, COLUMNS, FEATURE, RESPONSE)
        except (ValueError, TypeError):
            rejected.append(name)
        else:
            raise ValueError('Invalid saved state accepted: '+name)
    reject('unknown_format', lambda s:s.update(format='unknown'))
    reject('extra_executable_field', lambda s:s.update(import_path='forbidden'))
    reject('reordered_features', lambda s:s.update(feature_columns=list(reversed(COLUMNS))))
    reject('response_coordinate', lambda s:s.update(response_coordinate='wrong'))
    reject('missing_weight', lambda s:s['weights'].pop())
    reject('wrong_weight_shape', lambda s:s['weights'][0].pop())
    reject('nonfinite_weight', lambda s:s['weights'][0][0].__setitem__(0, float('nan')))
    reject('overflowing_weight', lambda s:s['weights'][0][0].__setitem__(0, 1e100))
    reject('boolean_weight', lambda s:s['weights'][0][0].__setitem__(0, True))
    reject('string_weight', lambda s:s['weights'][0][0].__setitem__(0, '1'))
    with tempfile.TemporaryDirectory(prefix='sciona-osic-state-') as temporary:
        directory = Path(temporary)
        (directory/'state.json').write_text(encoded)
        np.savez(directory/'queries.npz', queries=queries, expected=expected)
        hashes = {p.name:sha(p) for p in directory.iterdir()}
        process = subprocess.run([str(ROOT/'.venv/bin/python'), str(Path(__file__)), '--replay-directory', str(directory)],
            cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True, check=True, timeout=120)
        replay = json.loads(process.stdout)
        if hashes != {p.name:sha(p) for p in directory.iterdir()}:
            raise ValueError('Replay modified private state or inputs')
    if before != {str(p.relative_to(ROOT)):sha(p) for p in paths}:
        raise ValueError('Implementation changed during replay qualification')
    return dict(format='osic-quantile-state-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, training_updates=20, fresh_process_replay=replay,
        corruption_cases_rejected=rejected, private_artifacts_unchanged=True, software_sha256=before,
        runtime=dict(tensorflow=tf.__version__, tf_keras=keras.__version__, numpy=np.__version__),
        resource_policy=dict(native_threads=1, serial_training_and_replay=True),
        limitations=['Numeric inference state only; optimizer state and training continuation are not serialized.',
            'Twenty synthetic updates exercise learned-state replay; full training-schedule equivalence is documented separately.',
            'Caller must provide identically prepared ordered features and matching response units; coordinate identifiers are declarations, not proof.',
            'Feature preparation, image training, complete graph execution and catalog publication remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-directory', type=Path)
    args = parser.parse_args()
    report = child(args.replay_directory) if args.replay_directory else validate()
    if not args.replay_directory:
        (ROOT/'docs/reviews/osic_quantile_state_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report if args.replay_directory else dict(passed=True,
        corruption_cases=len(report['corruption_cases_rejected']), fresh_process_replay=report['fresh_process_replay'])))
