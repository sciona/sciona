"""Qualify connected preparation, complete fold fitting and inference replay."""
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
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pandas as pd

from sciona.osic_quantile_features import FIELDS
from sciona.osic_quantile_training import train_quantile_branch, replay_quantile_branch
from sciona.atoms.ml import tensorflow_distribution_losses
from sciona.atoms.numerical.statistics import ordered_sum

ROOT = Path(__file__).resolve().parents[1]
FEATURE = 'synthetic-joint-features.v1'
RESPONSE = 'synthetic-response.v1'
EVIDENCE = {
    'osic_quantile_features_validation.json': 'bc56e3e4',
    'osic_quantile_model_validation.json': '6d114c56',
    'osic_quantile_state_validation.json': '6d114c56',
    'osic_prediction_validation.json': '7449dbae',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence():
    result = {}
    for name, commit in EVIDENCE.items():
        path = ROOT/'docs/reviews'/name
        raw = subprocess.check_output(['git', 'show', commit+':'+str(path.relative_to(ROOT))], cwd=ROOT)
        if raw != path.read_bytes():
            raise ValueError('Prior qualification changed')
        report = json.loads(raw)
        if report['passed'] is not True:
            raise ValueError('Passing prior qualification required')
        for relative, digest in report['software_sha256'].items():
            target = ROOT.parent/relative if relative.startswith(('sciona-matcher/', 'sciona-atoms/')) else ROOT/relative
            if sha(target) != digest:
                raise ValueError('Prior qualified implementation changed')
        result[name] = dict(commit=commit, sha256=sha(path))
    return result


def child(directory):
    def forbidden(*args, **kwargs):
        raise AssertionError('Fitting forbidden in connected branch replay')
    keras.Model.fit = forbidden
    keras.Model.train_on_batch = forbidden
    keras.optimizers.legacy.Adam.apply_gradients = forbidden
    state = json.loads((directory/'state.json').read_text())
    with np.load(directory/'query.npz', allow_pickle=False) as arrays:
        actual = replay_quantile_branch(state['folds'], arrays['features'], state['columns'], FEATURE, RESPONSE,
                                         arrays['mask'], arrays['values'])
        np.testing.assert_array_equal(actual, arrays['expected'])
        return dict(passed=True, queries=len(actual), prediction_values=int(actual.size), exact=True, fitting_disabled=True)


def validate():
    previous = evidence()
    files = [Path(__file__), ROOT/'sciona/osic_quantile_training.py']
    files = {str(p.relative_to(ROOT)): p for p in files}
    for module in (tensorflow_distribution_losses, ordered_sum):
        files[module.__name__] = Path(inspect.getfile(module))
    before = {name:sha(path) for name,path in files.items()}
    rows = []
    for entity in range(135):
        for time in (0., 2.):
            rows.append([f'group-{entity}', time, 10.+entity/50-time*.1,
                         20.+entity%23, 'a'+str(entity%2), 'b'+str(entity%3)])
    rows += [['discarded', 4., 1., 35., 'a0', 'b0'], ['discarded', 4., 2., 35., 'a0', 'b0']]
    training = pd.DataFrame(rows, columns=FIELDS)
    baseline = pd.DataFrame([rows[i*2] for i in range(6)], columns=FIELDS)
    queries = pd.DataFrame([[f'group-{i}', time] for i in range(6) for time in (0., 6.)], columns=['entity', 'time'])
    originals = [f.copy(deep=True) for f in (training, baseline, queries)]
    result = train_quantile_branch(training, baseline, queries, feature_coordinate=FEATURE,
        response_coordinate=RESPONSE, seeds=(791, 792), allow_transductive=True,
        progress=lambda message:print(json.dumps(message), flush=True))
    for record in result['folds']:
        expected = dict(epochs=600, batch_size=128, training_rows=135, validation_rows=135,
                        optimizer_updates=1200, validation_checks=600)
        if any(record[k] != v for k, v in expected.items()):
            raise ValueError('Complete multibatch source schedule required')
    if result['oof_predictions'].shape != (270, 3) or not np.isfinite(result['oof_predictions']).all():
        raise ValueError('Complete out-of-fold predictions required')
    np.testing.assert_array_equal(result['predictions'][result['observed_mask'], 0],
                                  result['observed_values'][result['observed_mask']])
    np.testing.assert_array_equal(result['predictions'][result['observed_mask'], 1], .1)
    for current, original in zip((training, baseline, queries), originals):
        pd.testing.assert_frame_equal(current, original)
    with tempfile.TemporaryDirectory(prefix='sciona-osic-branch-') as temporary:
        directory = Path(temporary)
        (directory/'state.json').write_text(json.dumps(dict(folds=result['fold_states'], columns=result['feature_columns']), allow_nan=False))
        np.savez(directory/'query.npz', features=result['query_features'], mask=result['observed_mask'],
                 values=result['observed_values'], expected=result['predictions'])
        hashes = {p.name:sha(p) for p in directory.iterdir()}
        process = subprocess.run([str(ROOT/'.venv/bin/python'), str(Path(__file__)), '--replay-directory', str(directory)],
            cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True, check=True, timeout=120)
        replay = json.loads(process.stdout)
        if hashes != {p.name:sha(p) for p in directory.iterdir()}:
            raise ValueError('Replay changed saved state or input arrays')
    if previous != evidence() or before != {name:sha(path) for name,path in files.items()}:
        raise ValueError('Qualification changed during connected execution')
    return dict(format='osic-quantile-branch-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, folds=result['folds'], total_candidate_updates=2400,
        feature_count=len(result['feature_columns']), training_rows_after_deduplication=270,
        out_of_fold_predictions=270, known_observation_overrides=int(result['observed_mask'].sum()),
        fresh_process_replay=replay, source_component_evidence=previous, software_sha256=before,
        private_artifacts_unchanged=True, inputs_unchanged=True,
        runtime=dict(tensorflow=tf.__version__, tf_keras=keras.__version__, numpy=np.__version__, pandas=pd.__version__),
        resource_policy=dict(native_threads=1, serial_folds=True, data_loader_workers=0),
        limitations=['Complete quantile branch on synthetic mapped tables, not the complete two-branch competition workflow.',
            'Fresh-process replay consumes frozen prepared query features; a raw-table preprocessing state codec remains separate work.',
            'Historical random draws, population size and competitive accuracy are not reproduced.',
            'Joint query-dependent preparation is preserved; this is not a leakage-free validation estimate.',
            'The image branch, final connected blend, compositional catalog graph and publication gates remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-directory', type=Path)
    args = parser.parse_args()
    report = child(args.replay_directory) if args.replay_directory else validate()
    if not args.replay_directory:
        (ROOT/'docs/reviews/osic_quantile_branch_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report if args.replay_directory else dict(passed=True, updates=report['total_candidate_updates'],
        fresh_process_replay=report['fresh_process_replay'], approved=False)))
