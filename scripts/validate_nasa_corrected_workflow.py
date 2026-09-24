"""All-slot synthetic corrected NASA feature-to-prediction integration.

Source fitting, joins and prediction arithmetic are executed as pinned AST.
Corrected adapters supply feature tables; file IO stays in memory. No approval.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata as metadata
import io
import inspect
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
import xgboost as xgb

from sciona.nasa_feature_adapters import (estimated_departure_features,
    arrival_history_features, fit_airline_vocabulary, airline_features, adaptive_history_statistics)
from scripts.validate_nasa_third_training_sequence import SOURCE_HASH, INFERENCE_HASH
from sciona.atoms.ml.model_selection.population_masks import grouped_holdout_masks, residual_threshold_mask
from sciona.tabular_contracts import stable_feature_order, ordered_feature_matrix

ROOT = Path(__file__).resolve().parents[1]


def source_programs(source):
    trees = []
    for name, expected in [('Train_Models.source.py', SOURCE_HASH), ('Run_Inference.source.py', INFERENCE_HASH)]:
        raw = (source/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Pinned source program required')
        trees.append(ast.parse(raw))
    controls = {}
    for node in trees[0].body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'list_airports', 'bad_airports', 'mae_thresh_bad', 'mae_thresh_good'}:
                controls[name] = ast.literal_eval(node.value)
    class StableOrder(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node) == 'list(set(feature_cols))':
                return ast.copy_location(ast.Call(func=ast.Name(id='stable_feature_order', ctx=ast.Load()),
                    args=[ast.Name(id='feature_cols', ctx=ast.Load())], keywords=[]), node)
            return self.generic_visit(node)

    programs, hashes = [], []
    for tree, endpoint in zip(trees, ['median_overestimation', 'df_predict']):
        loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'airport')
        nodes = []
        for node in loop.body:
            names = [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else []
            call = node.value if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) else None
            if 'model_file_name' in names or (call is not None and ast.unparse(call.func) in {'print', 'pickle.dump'}):
                continue
            nodes.append(node)
            if endpoint == 'median_overestimation' and endpoint in names:
                break
            if endpoint == 'df_predict' and isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Subscript) and ast.unparse(t) == "df_predict['minutes_until_pushback']" for t in node.targets):
                break
        else:
            raise ValueError('Source execution boundary missing')
        module = ast.Module(body=nodes, type_ignores=[])
        # Explicit corrected behavior: preserve first-occurrence feature order.
        # All estimator controls, populations and numerical statements stay intact.
        module = ast.fix_missing_locations(StableOrder().visit(module))
        hashes.append(hashlib.sha256(ast.dump(module).encode()).hexdigest())
        programs.append(compile(module, '<corrected-feature-source-model-sequence>', 'exec'))
    return controls, programs, hashes


def synthetic_tables(rng, airport, *, training, vocabulary=None):
    anchor = pd.Timestamp('2000-01-01') + pd.Timedelta(days=0 if training else 3)
    groups, repeats = (120, 4) if training else (64, 1)
    queries = pd.DataFrame([dict(gufi=f'{"A" if training or i%2 else "U"}{i%30:02}-{i}.SRC.{airport[-3:]}',
        timestamp=anchor+pd.Timedelta(hours=6*j), airport=airport)
        for i in range(groups) for j in range(repeats)])
    estimates = []
    for row in queries.itertuples():
        for age in [10, 5]:
            estimates.append(dict(gufi=row.gufi, timestamp=row.timestamp-pd.Timedelta(minutes=age),
                departure_runway_estimated_time=row.timestamp+pd.Timedelta(minutes=float(rng.uniform(10, 80)))))
    etd = estimated_departure_features(queries, pd.DataFrame(estimates))
    stands, arrivals = [], []
    for j, query in enumerate(queries.timestamp.unique()):
        for i, age in enumerate([10, 20, 30]):
            actual = query-pd.Timedelta(minutes=age)
            identity = f'N{i:02}-{j}.SRC.{airport[-3:]}'
            stands.append(dict(gufi=identity, timestamp=actual, arrival_stand_actual_time=actual))
            arrivals.append(dict(gufi=identity, timestamp=actual,
                arrival_runway_actual_time=actual-pd.Timedelta(minutes=float(rng.uniform(5, 30)))))
    taxi = arrival_history_features(queries.timestamp, pd.DataFrame(stands), pd.DataFrame(arrivals), airport[-3:])
    vocabulary = fit_airline_vocabulary(queries) if vocabulary is None else vocabulary
    codes = airline_features(queries, vocabulary)
    if training:
        queries['minutes_until_pushback'] = 15+.8*etd.minutes_until_departure_from_timepoint.to_numpy()+rng.normal(0, 18, len(queries))
    return dict(labels=queries, etd=etd, taxi=taxi, codes=codes), vocabulary


def validate(source):
    dependencies = {
        'corrected_feature_adapters': ROOT/'sciona/nasa_feature_adapters.py',
        'adaptive_history_provider': Path(inspect.getsourcefile(adaptive_history_statistics)),
        'source_pin_definitions': ROOT/'scripts/validate_nasa_third_training_sequence.py',
        'population_masks_provider': Path(inspect.getsourcefile(grouped_holdout_masks)),
        'tabular_contracts': ROOT/'sciona/tabular_contracts.py',
    }
    dependency_pins = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in dependencies.items()}
    controls, programs, ast_hashes = source_programs(source)
    scenarios = []
    replay_digest = hashlib.sha256()
    for index, airport in enumerate(controls['list_airports']):
        rng = np.random.default_rng(9021+index)
        training, vocabulary = synthetic_tables(rng, airport, training=True)
        evaluation, _ = synthetic_tables(rng, airport, training=False, vocabulary=vocabulary)

        def loader(tables):
            def read(path, **kwargs):
                if 'train_labels_' in path or path.endswith('submission_data.csv'):
                    return tables['labels'].copy()
                for suffix, key in [('_etd.csv', 'etd'), ('_airlinecode.csv', 'codes'), ('_taxitime_to_gate.csv', 'taxi')]:
                    if path.endswith(suffix):
                        return tables[key].copy()
                raise ValueError('Unexpected source IO boundary')
            return read

        namespace = dict(np=np, pd=pd, xgb=xgb, GroupShuffleSplit=GroupShuffleSplit,
            mean_absolute_error=mean_absolute_error, mean_squared_error=mean_squared_error,
            airport=airport, raw_label_load_dir='', timepointgufi_root='', gufi_root='', timepoint_root='', **controls)
        namespace['stable_feature_order'] = stable_feature_order
        with threadpool_limits(limits=2), patch.object(pd, 'read_csv', loader(training)), contextlib.redirect_stdout(io.StringIO()):
            exec(programs[0], namespace)
        frame = namespace['df_data']
        if len(frame) != len(training['labels']):
            raise ValueError('Corrected training feature coverage incomplete')
        train, held = namespace['train_index'], namespace['test_index']
        if set(frame.iloc[train].gufi) & set(frame.iloc[held].gufi):
            raise ValueError('Training group leakage')
        _, group_codes = np.unique(frame.gufi.to_numpy(), return_inverse=True)
        train_mask, held_mask = grouped_holdout_masks(group_codes.astype(np.int64), .4, 42)
        np.testing.assert_array_equal(np.flatnonzero(train_mask), train)
        np.testing.assert_array_equal(np.flatnonzero(held_mask), held)
        offsets = {key: namespace[key] for key in ['median_underestimation', 'median_overestimation']}
        if not np.isfinite(list(offsets.values())).all():
            raise ValueError('Undefined source calibration offsets')
        cutoff = controls['mae_thresh_bad'] if airport in controls['bad_airports'] else controls['mae_thresh_good']
        retained = namespace['df_internal_test_lower']
        if retained.empty or not (retained.mae <= cutoff).all() or len(namespace['X_test']) != len(held):
            raise ValueError('Source training populations differ')
        held_frame = namespace['df_internal_test']
        selected_mask = residual_threshold_mask(held_frame.minutes_until_pushback.to_numpy(dtype=np.float64),
            held_frame.y_pred.to_numpy(dtype=np.float64), float(cutoff))
        np.testing.assert_array_equal(held_frame.index[selected_mask], retained.index)
        models = dict(regressor=namespace['regressor_lower'], classifier=namespace['estimate_classifier'], classifier_params=offsets)
        ordered = stable_feature_order(namespace['feature_cols'])
        if list(models['regressor'].get_booster().feature_names) != ordered:
            raise ValueError('Training feature order differs from declared order')
        ordered_feature_matrix(frame, ordered)

        def encode(frame, inferred_vocabulary):
            if set(inferred_vocabulary) != set(vocabulary):
                raise ValueError('Trained vocabulary inference differs')
            codes = airline_features(frame, vocabulary).drop(columns='airport')
            return frame.merge(codes, on='gufi', how='left', validate='many_to_one')

        outputs = []
        for candidate in [models, pickle.loads(pickle.dumps(models))]:
            scope = dict(np=np, pd=pd, airport=airport, model={airport: candidate}, debug=False,
                raw_label_load_dir='', timepointgufi_root_submission='', timepoint_root_submission='', grab_airlinecodes=encode)
            with threadpool_limits(limits=2), patch.object(pd, 'read_csv', loader(evaluation)), contextlib.redirect_stdout(io.StringIO()):
                exec(programs[1], scope)
            output = scope['df_predict']
            pd.testing.assert_frame_equal(output[['gufi', 'timestamp', 'airport']], evaluation['labels'])
            if len(output) != 64 or output.minutes_until_pushback.dtype != np.int32:
                raise ValueError('Final output contract differs')
            if output[scope['all_trained_features']].isna().any().any():
                raise ValueError('Corrected inference feature coverage incomplete')
            np.testing.assert_array_equal(ordered_feature_matrix(output, ordered),
                ordered_feature_matrix(output[output.columns[::-1]], ordered))
            predicted, probability = scope['y_pred_lower'], scope['y_prob_estimate']
            np.testing.assert_array_equal(scope['X_test_lower'].pred_minutes_until_pushback, predicted)
            expected = np.where(probability > .5, predicted+np.float32(offsets['median_underestimation']), predicted)
            expected = np.where(probability < .5, expected-np.float32(offsets['median_overestimation']), expected)
            np.testing.assert_array_equal(output.minutes_until_pushback, np.int32(np.around(expected)))
            outputs.append(output.minutes_until_pushback.to_numpy())
        np.testing.assert_array_equal(*outputs)
        replay_digest.update(json.dumps(ordered).encode())
        replay_digest.update(outputs[0].tobytes())
        scenario = dict(slot=index, cutoff=cutoff, input_rows=len(training['labels']),
            training_rows=len(train), held_rows=len(held), filtered_regression_rows=len(retained),
            predictions=64, complete_feature_coverage=True, disjoint_training_groups=True,
            reusable_population_masks_match_source=True,
            stable_feature_order_verified=True,
            final_int32_verified=True, same_environment_model_roundtrip_exact=True)
        scenarios.append(scenario)
        print(json.dumps(scenario), flush=True)
    if dependency_pins != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in dependencies.items()}:
        raise ValueError('Execution dependencies changed during validation')
    return dict(format='nasa-corrected-workflow.v1', passed=True, approved=False, synthetic_only=True,
        catalog_mutations=0, airport_slots=len(scenarios), scenarios=scenarios, source_ast_sha256=ast_hashes,
        source_sha256=dict(training=SOURCE_HASH, inference=INFERENCE_HASH),
        dependency_source_sha256=dependency_pins,
        synthetic_replay_signature=replay_digest.hexdigest(),
        implementation_sha256=hashlib.sha256((ROOT/'sciona/nasa_feature_adapters.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependencies={name: metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost']},
        limitations=['All ten source airport control slots executed with synthetic raw-frame inputs, not empirical competition performance.',
            'Corrected feature behavior is intentional; source feature defects are not reproduced by these adapters.',
            'Source set-derived feature order explicitly replaced with first-occurrence order; feature order is checked against fitted model metadata.',
            'File loaders and final file export replaced with in-memory tables; source model persistence replaced with same-environment pickle roundtrip.',
            'Current XGBoost environment only; historical backend and cross-process model interchange not qualified.',
            'Catalog graph construction, reusable decomposition of model fitting, environment/license review and publication gates remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_directory)
    (ROOT/'docs/reviews/competition_nasa_corrected_workflow.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'scenarios'}))
