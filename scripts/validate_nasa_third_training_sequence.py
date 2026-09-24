"""Run pinned numerical training statements on synthetic data with current XGBoost."""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata as metadata
import io
import json
import pickle
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = '2f8c7b7eb6c6c174128c37bc7d53c1db3236a9320699017f02387b406964fc03'
INFERENCE_HASH = '5d78b1dd59c14597deac54ce482ef08150eb6cc4dad99166cc8dfd7a49374c44'


def validate(source):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASH:
        raise ValueError('Pinned training source required')
    lines = raw.decode().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith('gss = GroupShuffleSplit('))
    end = next(i for i, line in enumerate(lines) if line.strip().startswith('median_overestimation = ('))
    tree = ast.parse(textwrap.dedent('\n'.join(lines[start:end+1])))
    # Suppress only model-file persistence and diagnostics; numerical statements stay unchanged.
    retained = []
    removed = []
    for node in tree.body:
        assignment = isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model_file_name' for t in node.targets)
        call = isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        persistence = call and ast.unparse(node.value.func) == 'pickle.dump'
        diagnostic = call and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print'
        if assignment or persistence or diagnostic:
            removed.append('model_path' if assignment else 'model_persistence' if persistence else 'diagnostic')
        else:
            retained.append(node)
    tree.body = retained
    code = compile(tree, '<pinned-synthetic-training>', 'exec')
    inference_raw = (source.parent / 'Run_Inference.source.py').read_bytes()
    if hashlib.sha256(inference_raw).hexdigest() != INFERENCE_HASH:
        raise ValueError('Pinned inference source required')
    inference_lines = inference_raw.decode().splitlines()
    first = next(i for i, line in enumerate(inference_lines) if line.strip().startswith('y_pred_lower = curr_model_regression_lower.predict('))
    last = next(i for i, line in enumerate(inference_lines) if line.strip().startswith("df_predict['minutes_until_pushback'] = np.int32"))
    inference_tree = ast.parse(textwrap.dedent('\n'.join(inference_lines[first:last+1])))
    inference_code = compile(inference_tree, '<pinned-synthetic-inference>', 'exec')
    cases = []
    for high_cutoff in [False, True]:
        rng = np.random.default_rng(3201)
        features = rng.normal(size=(480, 3))
        target = 50 + 25*features[:, 0] - 8*features[:, 1] + rng.normal(0, 18, 480)
        # Source software field names, populated only with constructed values.
        frame = pd.DataFrame(features, columns=['feature_a', 'feature_b', 'feature_c'])
        frame['minutes_until_pushback'] = target
        frame['gufi'] = np.repeat(np.arange(120), 4)
        namespace = dict(np=np, xgb=xgb, GroupShuffleSplit=GroupShuffleSplit,
            mean_absolute_error=mean_absolute_error, mean_squared_error=mean_squared_error,
            df_data=frame, feature_cols=['feature_a', 'feature_b', 'feature_c'],
            label_col=['minutes_until_pushback'], airport='synthetic_group',
            bad_airports=['synthetic_group'] if high_cutoff else [], mae_thresh_bad=30, mae_thresh_good=20)
        with threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
        train, held = namespace['train_index'], namespace['test_index']
        if set(frame.iloc[train]['gufi']) & set(frame.iloc[held]['gufi']):
            raise ValueError('Group leakage')
        low = namespace['df_internal_test_lower']
        cutoff = 30 if high_cutoff else 20
        if not len(low) or not (low['mae'] <= cutoff).all():
            raise ValueError('Residual filter failed')
        classifier_rows = namespace['X_test']
        if len(classifier_rows) != len(held):
            raise ValueError('Classifier population differs')
        np.testing.assert_array_equal(classifier_rows['label_estimation'],
            (classifier_rows['pred_minutes_until_pushback'] < classifier_rows['actual_minutes_until_pushback']).astype(int))
        offsets = [namespace['median_underestimation'], namespace['median_overestimation']]
        if not np.isfinite(offsets).all():
            raise ValueError('Undefined conditional correction')
        evaluation = pd.DataFrame(rng.normal(size=(64, 3)), columns=['feature_a', 'feature_b', 'feature_c'])
        outputs = []
        models = (namespace['regressor_lower'], namespace['estimate_classifier'])
        for regression, classifier in [models, pickle.loads(pickle.dumps(models))]:
            inference = dict(np=np, df_predict=evaluation.copy(), curr_model_regression_lower=regression,
                estimate_classifier=classifier, all_trained_features=list(regression.get_booster().feature_names),
                estimate_classifier_params=dict(median_underestimation=offsets[0], median_overestimation=offsets[1]))
            with threadpool_limits(limits=2):
                exec(inference_code, inference)
            predicted = inference['y_pred_lower']
            np.testing.assert_array_equal(inference['X_test_lower']['pred_minutes_until_pushback'], predicted)
            if not np.any(predicted != np.around(predicted)):
                raise ValueError('Synthetic inference did not exercise fractional predictions')
            probability = inference['y_prob_estimate']
            # pandas scalar arithmetic retains the prediction Series' float32 dtype.
            # NumPy 2 array + np.float64 scalar instead promotes, so make that cast explicit.
            if predicted.dtype != np.float32 or inference['y_pred'].dtype != np.float32:
                raise ValueError('Source prediction arithmetic dtype differs')
            expected = np.where(probability > .5, predicted + np.float32(offsets[0]), predicted)
            expected = np.where(probability < .5, expected - np.float32(offsets[1]), expected)
            np.testing.assert_array_equal(inference['y_pred'], expected)
            final = inference['df_predict']['minutes_until_pushback'].to_numpy()
            np.testing.assert_array_equal(final, np.int32(np.around(expected)))
            if final.dtype != np.int32:
                raise ValueError('Final source output dtype differs')
            outputs.append(final)
        np.testing.assert_array_equal(*outputs)
        cases.append(dict(cutoff=cutoff, training_rows=len(train), second_population_rows=len(held),
                          retained_regression_rows=len(low), classifier_rows=len(classifier_rows), finite_offsets=True,
                          inference_rows=64, unrounded_classifier_input_verified=True, final_int32_verified=True,
                          same_environment_pickle_roundtrip_exact=True))
    return dict(format='nasa-third-training-sequence.v1', passed=True, synthetic_only=True, approved=False,
        source_sha256=SOURCE_HASH, executed_ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        inference_source_sha256=INFERENCE_HASH, inference_ast_sha256=hashlib.sha256(ast.dump(inference_tree).encode()).hexdigest(),
        omitted_statements=removed, scenarios=cases, dependencies={n: metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'xgboost']},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), catalog_mutations=0,
        limitations=['Numerical source sequence executed with installed XGBoost 3.4.1, not historical 1.7.5 parity.',
                     'Synthetic feature matrix only; raw domain feature generation and complete submission workflow remain unqualified.',
                     'Inference recovers feature order from the trained regressor as in source; cross-process model interchange remains unqualified.',
                     'Classifier fit and offset estimation reuse the second population; no held-out accuracy claim.',
                     'Source model-file persistence and print diagnostics omitted; only same-environment in-memory pickle roundtrip qualified.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source)
    (ROOT / 'docs/reviews/competition_nasa_third_training_sequence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
