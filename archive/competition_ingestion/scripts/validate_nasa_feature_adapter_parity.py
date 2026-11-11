"""Compare corrected adapters with source kernels on their common contract."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sciona.nasa_feature_adapters import estimated_departure_features, arrival_history_features

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = 'cbcbe972af0d63a240923570e840744d573d972ee0bce19e04bf55a2c8b01444'


def validate(source):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASH:
        raise ValueError('Pinned source helper required')
    names = {'extract_etdv3', 'extract_taxi_to_gate_time'}
    functions = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(functions) != len(names):
        raise ValueError('Source kernel inventory differs')
    scope = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned-source-kernels>', 'exec'), scope)
    rng = np.random.default_rng(9471)
    anchor = pd.Timestamp('2000-01-01')
    history_errors = []
    for _ in range(64):
        count = int(rng.integers(1, 30))
        ages = rng.integers(-5, 221, count)
        future = rng.integers(-5, 91, count)
        estimates = pd.DataFrame(dict(gufi=['AAA.X.ZZZ']*count,
            timestamp=[anchor-pd.Timedelta(minutes=int(age)) for age in ages],
            departure_runway_estimated_time=[anchor+pd.Timedelta(minutes=int(value)) for value in future]))
        queries = pd.DataFrame(dict(gufi=['AAA.X.ZZZ'], timestamp=[anchor]))
        expected = scope['extract_etdv3']('AAA.X.ZZZ', [anchor], estimates.sort_values('timestamp', kind='stable'))
        actual = estimated_departure_features(queries, estimates)
        if expected:
            pd.testing.assert_frame_equal(actual, pd.DataFrame(expected)[actual.columns], check_dtype=False,
                                          check_exact=False, rtol=1e-13, atol=1e-13)
        elif not actual.empty:
            raise ValueError('Source empty ETD query differs')

        ages = rng.integers(1, 221, count)
        durations = rng.uniform(1, 60, count)
        # Whole nanoseconds are the common adapter clock. Compare durations after
        # timestamp construction, so rounding by pandas is shared with the source.
        events = [anchor-pd.Timedelta(minutes=int(age)) for age in ages]
        runway = [event-pd.Timedelta(minutes=float(duration)) for event, duration in zip(events, durations)]
        identities = [f'N{i:02}.X.ZZZ' for i in range(count)]
        stands = pd.DataFrame(dict(gufi=identities, timestamp=events, arrival_stand_actual_time=events))
        arrivals = pd.DataFrame(dict(gufi=identities, timestamp=events, arrival_runway_actual_time=runway))
        source_history = pd.DataFrame(dict(arrival_stand_actual_time=events,
            taxitime_to_gate=(stands.arrival_stand_actual_time-arrivals.arrival_runway_actual_time).dt.total_seconds()/60))
        expected = scope['extract_taxi_to_gate_time'](anchor, source_history)
        actual = arrival_history_features([anchor], stands, arrivals, 'ZZZ')
        if expected[0] == 0:
            assert actual.empty
        else:
            result = actual.iloc[0, 1:].to_numpy(dtype=float)
            np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
            history_errors.append(float(np.max(np.abs(result-expected))))
    return dict(format='nasa-feature-adapter-parity.v1', passed=True, approved=False,
        synthetic_only=True, source_kernel_cases=128, catalog_mutations=0,
        source_sha256=SOURCE_HASH, maximum_history_absolute_error=max(history_errors, default=0.),
        implementation_sha256=hashlib.sha256((ROOT/'sciona/nasa_feature_adapters.py').read_bytes()).hexdigest(),
        tests_sha256=hashlib.sha256((ROOT/'tests/test_nasa_feature_adapters.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changes_from_source=['All eligible ETD groups processed, including singleton and final group.',
            'Fixed training vocabulary, lexical frequency ties and always-present fallback category.',
            'Both observation streams filtered strictly before the query; latest complete snapshot per entity replaces source many-to-many joins.',
            'Ambiguous revisions, negative completed durations and invalid adapter inputs rejected explicitly.'],
        limitations=['Kernel comparisons use unique, already-available observations and the same timestamp clock; they do not establish whole-source equivalence.',
            'Corrections are intentional behavior changes; original source replay evidence remains separate.',
            'Domain parsing and feature joins are adapters; history computation delegates to the reusable adaptive-history atom.',
            'Complete model workflow, catalog graph bindings, licensing/environment review and publication gates remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source)
    (ROOT/'docs/reviews/competition_nasa_feature_adapter_parity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
