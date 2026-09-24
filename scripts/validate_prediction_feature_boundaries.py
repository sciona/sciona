"""Synthetic comparisons for source prediction-to-classifier boundaries."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sciona.atoms.ml.calibration import prediction_features as provider
from scripts.validate_nasa_third_training_sequence import SOURCE_HASH

ROOT = Path(__file__).resolve().parents[1]


def validate(source):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASH:
        raise ValueError('Pinned training software required')
    loop = next(n for n in ast.parse(raw).body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'airport')
    targets = {"X_test['label_estimation']",
        "X_test.loc[X_test['pred_minutes_until_pushback'] < X_test['actual_minutes_until_pushback'], 'label_estimation']"}
    labels = [n for n in loop.body if isinstance(n, ast.Assign)
              and any(ast.unparse(t) in targets for t in n.targets)]
    if len(labels) != 2:
        raise ValueError('Source label statements differ')
    module = ast.Module(body=labels, type_ignores=[])
    code = compile(module, '<pinned-source-label-statements>', 'exec')
    rng = np.random.default_rng(9462)
    for _ in range(128):
        prediction = rng.normal(0, 30, 64).astype(np.float32)
        observed = rng.normal(0, 30, 64)
        expected = np.int32(np.around(prediction, decimals=0))
        np.testing.assert_array_equal(provider.round_to_int32(prediction), expected)
        scope = dict(X_test=pd.DataFrame(dict(pred_minutes_until_pushback=expected,
            actual_minutes_until_pushback=observed)))
        exec(code, scope)
        np.testing.assert_array_equal(provider.underestimation_labels(observed, expected), scope['X_test'].label_estimation)
        features = rng.normal(size=(64, 3))
        for values in [expected, prediction]:
            frame = pd.DataFrame(features)
            frame['prediction'] = values
            np.testing.assert_array_equal(provider.append_prediction_feature(features, values), frame.to_numpy(dtype=np.float64))
    return dict(passed=True, approved=False, synthetic_only=True, catalog_mutations=0,
        source_sha256=SOURCE_HASH, source_label_ast_sha256=hashlib.sha256(ast.dump(module).encode()).hexdigest(),
        synthetic_cases=128, rounding_comparisons=128, label_comparisons=128,
        training_and_inference_feature_comparisons=256,
        provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Primitive composition boundaries only; full graph assembly remains pending.',
            'Source in-range rounding and strict label statements compared; overflow is explicitly rejected.',
            'Training prediction rounding and unrounded inference feature semantics remain separate.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source)
    (ROOT/'docs/reviews/nasa_prediction_feature_boundaries.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
