"""Replay the quantile branch from raw mapped queries and private numeric state."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
             'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
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
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pandas as pd

from sciona import osic_quantile_features as features
from sciona import osic_quantile_training as training
from sciona import osic_feature_state as feature_state
from sciona.osic_feature_state import export_preparation_state, transform_queries
from scripts.validate_osic_quantile_features import synthetic_tables
from sciona.atoms.ml import tensorflow_distribution_losses
from sciona.atoms.numerical.statistics import ordered_sum

ROOT = Path(__file__).resolve().parents[1]
FEATURE, RESPONSE = 'synthetic-joint-features.v1', 'synthetic-response.v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(directory):
    def forbidden(*args, **kwargs):
        raise AssertionError('Feature fitting and model training are forbidden during replay')
    keras.Model.fit = forbidden
    keras.Model.train_on_batch = forbidden
    keras.optimizers.legacy.Adam.apply_gradients = forbidden
    features.prepare_features = forbidden
    features.assemble_populations = forbidden
    features.joint_population_features = forbidden
    feature_state.export_preparation_state = forbidden
    feature_state.assemble_populations = forbidden
    feature_state.joint_population_features = forbidden
    training.prepare_features = forbidden
    training.train_quantile_branch = forbidden
    training.train_prepared_quantile_folds = forbidden
    bundle = json.loads((directory/'state.json').read_text())
    queries = pd.DataFrame(json.loads((directory/'queries.json').read_text()))
    prepared = transform_queries(bundle['preparation'], queries, feature_coordinate=FEATURE, response_coordinate=RESPONSE)
    columns = [json.dumps(c, separators=(',', ':')) for c in prepared['feature_columns']]
    prediction = training.replay_quantile_branch(bundle['models'], prepared['features'].astype(np.float32),
        columns, FEATURE, RESPONSE, prepared['observed_mask'], prepared['observed_values'])
    with np.load(directory/'expected.npz', allow_pickle=False) as expected:
        np.testing.assert_array_equal(prepared['features'], expected['features'])
        np.testing.assert_array_equal(prediction, expected['predictions'])
    return dict(passed=True, raw_queries=len(queries), feature_values=int(prepared['features'].size),
                features_exact=True, predictions_exact=True, feature_fitting_disabled=True, model_updates_disabled=True)


def validate():
    paths = [Path(__file__), ROOT/'sciona/osic_feature_state.py', ROOT/'sciona/osic_quantile_features.py',
        ROOT/'sciona/osic_quantile_training.py', ROOT/'sciona/osic_quantile_state.py',
        ROOT/'sciona/osic_quantile_model.py', ROOT/'sciona/osic_quantile_objective.py',
        ROOT/'sciona/tensorflow_distribution_losses.py', ROOT/'sciona/osic_predictions.py']
    paths = {str(p.relative_to(ROOT)): p for p in paths}
    for module in (tensorflow_distribution_losses, ordered_sum):
        paths[module.__name__] = Path(inspect.getfile(module))
    before = {name:sha(path) for name,path in paths.items()}
    train, baseline, queries = synthetic_tables()
    # Include known baseline queries as well as extrapolation on both sides.
    queries = pd.concat([queries, baseline[['entity', 'time']]], ignore_index=True)
    state = export_preparation_state(train, baseline, queries, feature_coordinate=FEATURE,
        response_coordinate=RESPONSE, allow_transductive=True)
    state = json.loads(json.dumps(state, allow_nan=False))
    fitted = features.prepare_features(train, baseline, queries, allow_transductive=True)
    original = copy.deepcopy(state)
    replay = transform_queries(state, queries, feature_coordinate=FEATURE, response_coordinate=RESPONSE)
    np.testing.assert_array_equal(replay['features'], fitted['populations']['query']['features'])
    # A second arithmetic dtype must survive the numeric codec unchanged.
    f32_train, f32_base, f32_query = train.copy(), baseline.copy(), queries.copy()
    for frame in (f32_train, f32_base, f32_query):
        for field in ('time','target','numeric'):
            if field in frame:
                frame[field] = frame[field].astype(np.float32)
    f32_state = json.loads(json.dumps(export_preparation_state(f32_train, f32_base, f32_query,
        feature_coordinate=FEATURE, response_coordinate=RESPONSE, allow_transductive=True), allow_nan=False))
    np.testing.assert_array_equal(transform_queries(f32_state, f32_query, feature_coordinate=FEATURE,
        response_coordinate=RESPONSE)['features'], features.prepare_features(f32_train, f32_base, f32_query,
        allow_transductive=True)['populations']['query']['features'])
    rejected = []
    def reject(label, mutation):
        changed = copy.deepcopy(state)
        mutation(changed)
        try:
            transform_queries(changed, queries, feature_coordinate=FEATURE, response_coordinate=RESPONSE)
        except (ValueError, TypeError):
            rejected.append(label)
        else:
            raise ValueError('Invalid preparation state accepted: '+label)
    reject('format', lambda s:s.update(format='unknown'))
    reject('executable_field', lambda s:s.update(import_path='forbidden'))
    reject('coordinate', lambda s:s.update(feature_coordinate='wrong'))
    reject('query_scope', lambda s:s['query_scope'][0].update(time=1234.))
    reject('feature_order', lambda s:s['feature_columns'].reverse())
    reject('missing_baseline', lambda s:s['observations'].pop())
    reject('duplicate_baseline', lambda s:s['observations'].append(copy.deepcopy(s['observations'][0])))
    reject('nonfinite_baseline', lambda s:s['observations'][0].update(baseline_target=float('nan')))
    reject('zero_range', lambda s:s['ranges'].update(numeric=[1., 1.]))
    reject('object_dtype', lambda s:s['dtypes'].update(numeric='object'))
    if state != original:
        raise ValueError('Replay mutated fitted state')
    result = training.train_quantile_branch(train, baseline, queries, feature_coordinate=FEATURE,
        response_coordinate=RESPONSE, seeds=(910, 911), allow_transductive=True)
    with tempfile.TemporaryDirectory(prefix='sciona-osic-raw-replay-') as temporary:
        directory = Path(temporary)
        (directory/'state.json').write_text(json.dumps(dict(preparation=state, models=result['fold_states']), allow_nan=False))
        (directory/'queries.json').write_text(queries.to_json(orient='records', double_precision=15))
        np.savez(directory/'expected.npz', features=fitted['populations']['query']['features'], predictions=result['predictions'])
        hashes = {p.name:sha(p) for p in directory.iterdir()}
        process = subprocess.run([str(ROOT/'.venv/bin/python'), str(Path(__file__)), '--replay-directory', str(directory)],
            cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True, check=True, timeout=120)
        replay_report = json.loads(process.stdout)
        if hashes != {p.name:sha(p) for p in directory.iterdir()}:
            raise ValueError('Replay changed private artifacts')
    if before != {name:sha(path) for name,path in paths.items()}:
        raise ValueError('Implementation changed during qualification')
    return dict(format='osic-raw-table-replay.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, fresh_process_replay=replay_report,
        fitted_dtype_cases=['float64','float32'], corruption_cases_rejected=rejected,
        model_training_epochs=[r['epochs'] for r in result['folds']],
        software_sha256=before, private_artifacts_unchanged=True, fitted_state_unchanged=True,
        runtime=dict(tensorflow=tf.__version__, tf_keras=keras.__version__, numpy=np.__version__, pandas=pd.__version__),
        resource_policy=dict(native_threads=1, serial_training_and_replay=True),
        limitations=['Replay is bound to the exact declared query batch and order; a new transductive query population requires new preparation and training.',
            'Saved preprocessing state includes caller-specific baselines and query scope and must remain private runtime material.',
            'Feature and model state validation is structural; authenticity requires external artifact integrity/provenance controls.',
            'Only the quantile branch is connected; image training, final two-branch graph, dependency qualification and catalog publication remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-directory', type=Path)
    args = parser.parse_args()
    result = child(args.replay_directory) if args.replay_directory else validate()
    if not args.replay_directory:
        (ROOT/'docs/reviews/osic_raw_table_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result if args.replay_directory else dict(passed=True,
        corruption_cases=len(result['corruption_cases_rejected']), replay=result['fresh_process_replay'])))
