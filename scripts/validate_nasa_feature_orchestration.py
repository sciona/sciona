"""Execute pinned NASA feature orchestration and training joins on synthetic inputs.

Only loaders, progress display and file output are replaced. Source computation,
including omissions and failure behavior, is preserved. No catalog promotion.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'helper.py': 'cbcbe972af0d63a240923570e840744d573d972ee0bce19e04bf55a2c8b01444',
    'Feature_Processing-ETD_no_PQDM.source.py': '02a1bd1e20c41ed5084a7363165211713c70291efa2695d7f45306801f1fdb70',
    'Feature_Processing-TaxiTimeToGate.source.py': '7f40bff888af396be3ca42879f1306a7f36407b205ecf17e257f82ee6c818b8e',
    'Feature_Processing-AirplaneCodes.source.py': '862049a092252f1a23683acaf7fbf080f438e4abc42ab545622863970c8ff098',
    'Train_Models.source.py': '2f8c7b7eb6c6c174128c37bc7d53c1db3236a9320699017f02387b406964fc03',
    'Run_Inference.source.py': '5d78b1dd59c14597deac54ce482ef08150eb6cc4dad99166cc8dfd7a49374c44',
}


def execute(nodes, namespace):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-source-statements>', 'exec'), namespace)


def validate(source):
    trees = {}
    for name, expected in PINS.items():
        raw = (source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Pinned software source differs: ' + name)
        trees[name] = ast.parse(raw)
    helpers = {'extract_etdv3', 'split_gufi', 'extract_taxi_to_gate_time'}
    functions = [n for n in trees['helper.py'].body if isinstance(n, ast.FunctionDef) and n.name in helpers]
    if len(functions) != len(helpers):
        raise ValueError('Helper inventory differs')
    base = dict(np=np, pd=pd, tqdm=lambda values, **_: values, airport='KZZZ',
                bool_submission_prep=False, load_dir='', sav_dir='')
    execute(functions, base)
    anchor = pd.Timestamp('2000-01-01')

    def inputs(count=26, future_observation=False):
        labels = pd.DataFrame([dict(gufi=f'C{i:02}.SRC.ZZZ', timestamp=anchor,
            airport='KZZZ', minutes_until_pushback=float(10 + i)) for i in range(count)])
        estimates = pd.DataFrame([dict(gufi=row.gufi,
            timestamp=anchor-pd.Timedelta(minutes=10),
            departure_runway_estimated_time=anchor+pd.Timedelta(minutes=20))
            for row in labels.itertuples()])
        stands, arrivals = [], []
        for i, (age, duration) in enumerate([(10, 10), (70, 20), (130, 30)]):
            identity = f'N{i:02}.SRC.ZZZ'
            actual = anchor-pd.Timedelta(minutes=age)
            observed = anchor+pd.Timedelta(minutes=5) if future_observation else actual
            stands.append(dict(gufi=identity, timestamp=observed, arrival_stand_actual_time=actual))
            arrivals.append(dict(gufi=identity, timestamp=observed,
                arrival_runway_actual_time=actual-pd.Timedelta(minutes=duration)))
        return labels, estimates, pd.DataFrame(stands), pd.DataFrame(arrivals)

    def feature(name, data):
        labels, estimates, stands, arrivals = data
        namespace = dict(base)
        namespace.update(data_loader_etd=lambda *_: (labels.copy(), estimates.copy()),
            data_loader_submission_etd=lambda *_: (labels.copy(), estimates.copy()),
            data_loader_train_labels=lambda *_: labels.copy(),
            data_loader_submission_train_labels=lambda *_: labels.copy(),
            data_loader=lambda *_: (pd.DataFrame(), estimates.copy(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame(), arrivals.copy(), pd.DataFrame(), stands.copy()))
        loop = [n for n in trees[name].body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'airport']
        if len(loop) != 1:
            raise ValueError('Expected one airport orchestration loop')
        captured = []
        def capture(frame, *args, **kwargs):
            captured.append(frame.copy(deep=True))
        with patch.object(pd.DataFrame, 'to_csv', capture):
            execute(loop[0].body, namespace)
        if len(captured) != 1:
            raise ValueError('Expected one feature output')
        return captured[0]

    etd_name = 'Feature_Processing-ETD_no_PQDM.source.py'
    taxi_name = 'Feature_Processing-TaxiTimeToGate.source.py'
    codes_name = 'Feature_Processing-AirplaneCodes.source.py'
    data = inputs()
    etd = feature(etd_name, data)
    assert len(etd) == 25 and set(etd.gufi) == set(data[0].gufi[:-1])
    np.testing.assert_array_equal(etd.minutes_until_departure_from_timepoint, np.full(25, 20.))
    np.testing.assert_array_equal(etd.minutes_until_departure_from_timestamp, np.full(25, 30.))
    assert feature(etd_name, inputs(1)).empty
    codes = feature(codes_name, data)
    assert len(codes) == 26 and codes['Other'].sum() == 1
    try:
        feature(codes_name, inputs(3))
    except KeyError as error:
        if 'Other' not in str(error):
            raise
    else:
        raise AssertionError('Expected source missing-Other failure')
    taxi = feature(taxi_name, data)
    np.testing.assert_array_equal(taxi[['found_counts_taxitime_to_gate', 'taxitime_to_gate_mean',
        'taxitime_to_gate_std']].to_numpy(), [[2., 15., 5.]])
    future_taxi = feature(taxi_name, inputs(future_observation=True))
    pd.testing.assert_frame_equal(taxi, future_taxi)

    # Original feature reads, merge types, dropna and feature selection, up to fitting.
    training_loop = next(n for n in trees['Train_Models.source.py'].body
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'airport')
    nodes = []
    for node in training_loop.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'gss' for t in node.targets):
            break
        nodes.append(node)
    else:
        raise ValueError('Training split boundary missing')
    namespace = dict(base, raw_label_load_dir='', timepointgufi_root='', gufi_root='', timepoint_root='')
    def read_synthetic(path, **kwargs):
        if 'train_labels_' in path or path.endswith('submission_data.csv'):
            return data[0].copy()
        if path.endswith('_etd.csv'):
            return etd.copy()
        if path.endswith('_airlinecode.csv'):
            return codes.copy()
        if path.endswith('_taxitime_to_gate.csv'):
            return taxi.copy()
        raise ValueError('Unexpected source read boundary')
    with patch.object(pd, 'read_csv', read_synthetic):
        execute(nodes, namespace)
    combined = namespace['df_data']
    assert len(combined) == 25 and set(combined.gufi) == set(etd.gufi)
    assert not combined.isna().any().any()
    selected = combined[list(set(namespace['feature_cols']))]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes)
    assert np.isfinite(selected.to_numpy(dtype=float)).all()

    # Model metadata supplies only the selected feature order. No inference is mocked.
    # Stop before predict: this qualifies the original inference joins, not a fit.
    names = list(selected.columns)
    metadata = SimpleNamespace(get_booster=lambda: SimpleNamespace(feature_names=names))
    inference = dict(base, model={'KZZZ': dict(regressor=metadata, classifier=None, classifier_params={})},
        raw_label_load_dir='', timepointgufi_root_submission='', timepoint_root_submission='', debug=False)
    functions = [n for n in trees['Run_Inference.source.py'].body
        if isinstance(n, ast.FunctionDef) and n.name in {'split_gufi', 'grab_airlinecodes'}]
    assert len(functions) == 2
    execute(functions, inference)
    loop = next(n for n in trees['Run_Inference.source.py'].body
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'airport')
    nodes = []
    for node in loop.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'y_pred_lower' for t in node.targets):
            break
        nodes.append(node)
    else:
        raise ValueError('Inference prediction boundary missing')
    with patch.object(pd, 'read_csv', read_synthetic):
        execute(nodes, inference)
    predicted_inputs = inference['df_predict']
    assert len(predicted_inputs) == 26 and not inference['missing_feats']
    assert predicted_inputs['minutes_until_departure_from_timepoint'].isna().sum() == 1
    assert set(predicted_inputs.gufi) == set(data[0].gufi)
    pd.testing.assert_frame_equal(predicted_inputs[names].iloc[:25].reset_index(drop=True),
                                  selected.reset_index(drop=True), check_dtype=False)
    return dict(format='nasa-feature-orchestration.v1', passed=True, approved=False,
        synthetic_only=True, catalog_mutations=0, source_sha256=PINS,
        feature_branches_executed=3, training_feature_joins_executed=True,
        synthetic_input_rows=26, training_rows_retained=25,
        inference_feature_joins_executed=True, inference_rows_retained=26,
        findings=dict(etd_final_group_omitted=True, etd_singleton_output_empty=True,
            categorical_output_with_rare_category=True, categorical_no_rare_category_raises=True,
            adaptive_history_statistics_match=True, future_observations_included=True,
            joined_features_numeric_and_finite=True, inference_retains_missing_etd_row=True,
            shared_training_inference_features_equal=True),
        dependency_versions=dict(numpy=np.__version__, pandas=pd.__version__),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            'Pinned serial feature branches and training/inference feature joins; file loaders replaced with synthetic frames and outputs captured in memory. Model metadata supplies feature names only; execution stops before inference.',
            'Passed means source behavior reproduced, including defects; it is not a publication qualification.',
            'Source serial ETD branch omits its final entity. Parallel helper also has a removed pandas append call.',
            'Source categorical feature export fails when no rare category creates Other.',
            'Arrival features include observations recorded after the query; historical availability must be enforced by an explicit upstream contract or a documented correction.',
            'Training inner join drops the omitted ETD entity; inference left join retains it with missing ETD features. No imputation was added.',
            'No full raw-file loading, all-airport model fitting or final submission coverage claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_directory)
    (ROOT / 'docs/reviews/competition_nasa_feature_orchestration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
