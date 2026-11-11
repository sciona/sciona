"""Compare independent lag operations with privately cached public software.

Only pinned numerical function definitions are evaluated, on synthetic arrays.
No source IO, training script, source records or weights are executed or copied.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.m5_lags import rolling, shifted


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(source_root):
    pins_path = ROOT / 'docs/reviews/competition_m5_source_pins.json'
    pins = json.loads(pins_path.read_text())
    source_name = '1-1. recursive_store_PREDICT.py'
    entries = [(name, digest) for name, digest in pins['files'].items() if Path(name).name == source_name]
    sources = list(source_root.rglob(source_name))
    if len(entries) != 1 or len(sources) != 1 or sha(sources[0]) != entries[0][1]:
        raise ValueError('Unique pinned source required')
    tree = ast.parse(sources[0].read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ('make_lag', 'make_lag_roll')]
    if {node.name for node in selected} != {'make_lag', 'make_lag_roll'} or len(selected) != 2:
        raise ValueError('Expected numerical source functions missing')
    numerical = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {'np': np, 'TARGET': 'value', '__builtins__': {'str': str}}
    exec(compile(numerical, '<pinned-numerical-functions>', 'exec'), namespace)
    rng = np.random.default_rng(903)
    count = 0
    for missing in (False, True):
        groups = np.tile([0, 1, 2], 250)
        values = rng.uniform(0, 32, len(groups))
        if missing:
            values[rng.choice(len(values), 11, replace=False)] = np.nan
        namespace['base_test'] = pd.DataFrame({'id': groups, 'd': np.arange(len(groups)), 'value': values})
        # The source mutates a frame slice; warnings contain no useful evidence.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
            for shift in range(28, 43):
                expected = namespace['make_lag'](shift).to_numpy()[:, 0]
                np.testing.assert_equal(shifted(values, groups, shift).astype(np.float16), expected)
                count += 1
            for shift in (1, 7, 14):
                for window in (7, 14, 30, 60):
                    expected = namespace['make_lag_roll']([shift, window]).to_numpy()[:, 0]
                    actual = rolling(values, groups, shift, window)
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
                    np.testing.assert_equal(actual.astype(np.float16), expected.astype(np.float16))
                    count += 1
    paths = [ROOT/'sciona/m5_lags.py', ROOT/'tests/test_m5_lags.py', Path(__file__).resolve(), pins_path]
    return dict(status='passed', approved=False, catalog_mutations=0,
        source_commit=pins['commit'], source_software_sha256=entries[0][1],
        checks=dict(source_function_comparisons=count, synthetic_only=True,
                    interleaved_groups=True, missing_values=True,
                    stored_float16=True, recursive_full_precision=True),
        sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
        scope='Independent lag and temporary rolling mean components only. Fixed-window sample deviation has a separate pandas oracle. Full M5 preprocessing, models and recursive forecasting remain unqualified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.source_root)
    (ROOT/'docs/reviews/competition_m5_lag_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['checks']))
