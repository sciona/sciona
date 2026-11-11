"""Synthetic row/order equivalence against the pinned winner population helper."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sciona.wheat_pseudo_population import build_pseudo_population


def main(source, output):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != '7498ce7dbfca13489c66dc515658240fad3156fb1e99ee2c01b56507f2bf5f3a':
        raise ValueError('population reference source differs')
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'make_pseudo_dataframe']
    if len(nodes) != 1:
        raise ValueError('source population helper absent')
    namespace = dict(np=np, pd=pd, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-wheat-population>', 'exec'), namespace)
    original = pd.DataFrame(dict(image_id=['synthetic_a', 'synthetic_a', 'synthetic_b', 'synthetic_c'],
        fold=[0, 0, 1, 1], isbox=[True, True, True, False],
        xmin=[1., 3., 2., np.nan], ymin=[1., 4., 2., np.nan],
        xmax=[8., 9., 7., np.nan], ymax=[8., 9., 7., np.nan]), index=[8, 12, 17, 23])
    query = pd.DataFrame({'image_id': ['synthetic_z', 'synthetic_q', 'synthetic_z']})
    predictions = {'synthetic_z': (np.array([[2., 3., 8., 9.], [5., 7., 11., 12.]]), np.array([.7, .8])),
                   'synthetic_q': (np.empty((0, 4)), np.empty(0))}
    previous_rng = np.random.get_state()
    original_writer = pd.DataFrame.to_csv
    try:
        for seed in range(16):
            captured = {}
            def capture(frame, name, **kwargs):
                if name not in ('train.csv', 'valid.csv') or kwargs != {'index': False}:
                    raise ValueError('unexpected reference write')
                captured[name] = frame.copy(deep=True)
            pd.DataFrame.to_csv = capture
            np.random.seed(seed)
            namespace['make_pseudo_dataframe'](query, predictions, 'synthetic_query',
                                               original.copy(deep=True), 'synthetic_train', 1)
            state_before = np.random.get_state()
            train, valid = build_pseudo_population(original, query['image_id'], predictions,
                train_directory='synthetic_train', query_directory='synthetic_query', seed=seed)
            pd.testing.assert_frame_equal(train, captured['train.csv'], check_exact=True)
            pd.testing.assert_frame_equal(valid, captured['valid.csv'], check_exact=True)
            state_after = np.random.get_state()
            assert np.array_equal(state_before[1], state_after[1]) and state_before[2:] == state_after[2:]
    finally:
        pd.DataFrame.to_csv = original_writer
        np.random.set_state(previous_rng)
    files = ['sciona/wheat_pseudo_population.py', 'scripts/validate_wheat_pseudo_population.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        shuffle_seeds_checked=16, training_and_validation_rows_exact=True,
        empty_query_negative_retained=True, empty_heldout_annotations_excluded=True,
        global_numpy_rng_unchanged=True, source_sha256=digest,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Synthetic population construction only; no detector predictions or pseudo-model training executed.',
                'Reference and reconstruction share installed pandas/NumPy; historical library identity remains separate.',
                'Unique original row index required; source duplicate-index side effects are rejected.',
                'Full pipeline and publication gates remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'implementation_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
