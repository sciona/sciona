"""Source-bound synthetic diagnosis of grouped feature orchestration."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HELPER_HASH = 'cbcbe972af0d63a240923570e840744d573d972ee0bce19e04bf55a2c8b01444'


def diagnose(source):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_HASH:
        raise ValueError('Pinned helper required')
    tree = ast.parse(raw)
    names = {'extract_etdv3', 'extract_etd_curr_airport', 'extract_taxi_to_gate_time'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(functions) != len(names):
        raise ValueError('Source function inventory differs')
    namespace = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned-feature-functions>', 'exec'), namespace)
    results = []
    original = namespace['extract_etdv3']
    for count in [1, 3]:
        anchor = pd.Timestamp('2000-01-01')
        labels = pd.DataFrame([dict(gufi=f'synthetic-{i}', timestamp=anchor) for i in range(count)])
        estimates = pd.DataFrame([dict(gufi=f'synthetic-{i}', timestamp=anchor-pd.Timedelta(minutes=10),
            departure_runway_estimated_time=anchor+pd.Timedelta(minutes=20)) for i in range(count)])
        calls = []
        def observe(identity, times, frame):
            calls.append(identity)
            return original(identity, times, frame)
        namespace.update(extract_etdv3=observe, data_loader_etd=lambda *_: (labels, estimates))
        failed_append = False
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                namespace['extract_etd_curr_airport']('', '', 'synthetic', False)
            except AttributeError as error:
                if 'append' not in str(error):
                    raise
                failed_append = True
        if not failed_append or calls != [f'synthetic-{i}' for i in range(count-1)]:
            raise ValueError('Expected pinned source behavior changed')
        results.append(dict(input_groups=count, processed_groups=len(calls), final_group_skipped=True,
                            missing_dataframe_append=True, output_written=False))
    # One-shot window widening: with zero events in 60 minutes, select 180 directly.
    anchor = pd.Timestamp('2000-01-01')
    history = pd.DataFrame(dict(arrival_stand_actual_time=[anchor-pd.Timedelta(minutes=m) for m in [70, 130]],
                                taxitime_to_gate=[10., 30.]))
    stats = namespace['extract_taxi_to_gate_time'](anchor, history)
    np.testing.assert_array_equal(stats, [2., 20., 10.])
    return dict(format='nasa-feature-extraction-diagnostic.v1', source_sha256=HELPER_HASH,
        diagnosis_verified=True, synthetic_only=True, approved=False, catalog_mutations=0,
        grouped_orchestration_cases=results, adaptive_window_case=dict(empty_initial_window=True,
            selected_lookback_minutes=180, events=2, population_standard_deviation=True),
        dependency_versions=dict(numpy=np.__version__, pandas=pd.__version__),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Only selected unmodified source functions executed with synthetic loader inputs.',
                    'No compatibility shim or source correction applied; complete raw-feature pipeline remains unqualified.',
                    'Final-group omission and missing append API must be accounted for explicitly in a faithful or corrected realization.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.source)
    (ROOT / 'docs/reviews/competition_nasa_feature_diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
