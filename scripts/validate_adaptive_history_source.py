"""Compare reusable adaptive statistics against the pinned numeric helper."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sciona.atoms.ml.calibration.adaptive_history as provider

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = 'cbcbe972af0d63a240923570e840744d573d972ee0bce19e04bf55a2c8b01444'


def validate(source):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASH:
        raise ValueError('Pinned helper required')
    functions = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'extract_taxi_to_gate_time']
    if len(functions) != 1:
        raise ValueError('Unique source function required')
    namespace = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned-adaptive-history>', 'exec'), namespace)
    rng = np.random.default_rng(5518)
    anchor = pd.Timestamp('2000-01-01')
    maximum = 0.
    empty = 0
    for index in range(128):
        n = index % 65
        times = rng.integers(-250, 3, size=n, dtype=np.int64)
        if n >= 4:
            times[:4] = [-180, -120, -60, 0]
        values = rng.normal(size=n).astype(np.float64)
        frame = pd.DataFrame(dict(arrival_stand_actual_time=anchor+pd.to_timedelta(times, unit='m'), taxitime_to_gate=values))
        expected = namespace['extract_taxi_to_gate_time'](anchor, frame)
        actual = np.asarray(provider.adaptive_history_statistics(times, values, 0, np.array([60,120,180], dtype=np.int64), 3))
        if actual[0] != expected[0]:
            raise ValueError('Source membership count differs')
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13, equal_nan=True)
        finite = np.isfinite(expected)
        maximum = max(maximum, float(np.max(np.abs(actual[finite]-expected[finite]))))
        empty += int(actual[0] == 0)
    examples = []
    for domain, values in [('manufacturing_measurements', [10., 30.]), ('energy_sensor_readings', [100., 300.])]:
        count, mean, std = provider.adaptive_history_statistics(np.array([-70,-130], dtype=np.int64),
            np.array(values), 0, np.array([60,120,180], dtype=np.int64), 3)
        if count != 2 or mean != sum(values)/2 or std != (values[1]-values[0])/2:
            raise ValueError('Synthetic domain example differs')
        examples.append(dict(domain=domain, executed=True))
    return dict(format='adaptive-history-source-comparison.v1', passed=True, synthetic_only=True, source_cases=128,
        empty_windows=empty, maximum_absolute_error=maximum, tolerance=dict(relative=1e-13, absolute=1e-13),
        synthetic_domains=examples, source_sha256=SOURCE_HASH,
        provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), approved=False, catalog_mutations=0,
        limitations=['Finite float64 values with int64 caller-defined time units; source null-value skipping is outside this contract.',
            'Empty selected populations return count zero and undefined statistics; no imputation.',
            'Synthetic examples verify transfer of the numerical operation, not empirical domain suitability.',
            'Atom validation only; complete NASA feature orchestration and publication qualification remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source)
    (ROOT / 'docs/reviews/adaptive_history_source_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
