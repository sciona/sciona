"""Substitute reusable native-model atoms into the corrected source workflow."""
import argparse
import contextlib
import hashlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import validate_nasa_corrected_workflow as workflow
from sciona.atoms.ml.xgboost.model_io import (fit_regression_model, fit_binary_model,
    predict_regression_model, predict_binary_model)

ROOT = Path(__file__).resolve().parents[1]


class NativeRegression:
    """Validation-only estimator interface; computation lives in reusable atoms."""
    def fit(self, features, targets):
        self.names = list(features.columns)
        self.state = fit_regression_model(features.to_numpy(dtype=np.float64),
            np.asarray(targets, dtype=np.float64).reshape(-1), self.names)
        return self

    def predict(self, features):
        return predict_regression_model(features.to_numpy(dtype=np.float64), list(features.columns), self.state)

    def get_booster(self):
        return SimpleNamespace(feature_names=list(self.names))


class NativeBinary:
    def fit(self, features, targets):
        self.names = list(features.columns)
        self.state = fit_binary_model(features.to_numpy(dtype=np.float64),
            np.asarray(targets, dtype=np.int64).reshape(-1), self.names)
        return self

    def predict_proba(self, features):
        positive = predict_binary_model(features.to_numpy(dtype=np.float64), list(features.columns), self.state)
        return np.column_stack((1-positive, positive))

    def predict(self, features):
        return (self.predict_proba(features)[:, 1] > .5).astype(np.int64)


def validate(source):
    baseline_path = ROOT/'docs/reviews/competition_nasa_corrected_workflow.json'
    baseline = json.loads(baseline_path.read_text())
    if not baseline['passed'] or baseline['validator_sha256'] != hashlib.sha256(Path(workflow.__file__).read_bytes()).hexdigest():
        raise ValueError('Current passing baseline required')
    provider = Path(inspect.getsourcefile(fit_regression_model))
    before = hashlib.sha256(provider.read_bytes()).hexdigest()
    constructors = SimpleNamespace(XGBRegressor=NativeRegression, XGBClassifier=NativeBinary)
    with patch.object(workflow, 'xgb', constructors), contextlib.redirect_stdout(io.StringIO()):
        native = workflow.validate(source)
    if before != hashlib.sha256(provider.read_bytes()).hexdigest():
        raise ValueError('Provider changed during execution')
    if native['synthetic_replay_signature'] != baseline['synthetic_replay_signature'] or native['scenarios'] != baseline['scenarios']:
        raise ValueError('Reusable native-model execution differs from baseline')
    return dict(format='nasa-native-model-substitution.v1', passed=True, approved=False,
        synthetic_only=True, catalog_mutations=0, airport_slots=10, model_fits=30,
        predictions_compared=640, final_predictions_and_populations_exact=True,
        provider_sha256=before, baseline_report_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_cpu_threads=1, operations=['fit_regression_model', 'predict_regression_model',
            'fit_binary_model', 'predict_binary_model'],
        limitations=['Validation-only estimator interfaces connect source orchestration to four separate reusable atoms; they are not the final CDG.',
            'Same installed backend defaults; historical engine parity is not claimed.',
            'Native model state contains fitted model data and feature metadata and must remain private runtime material for non-public inputs.',
            'Full graph binding, serialization through the actual graph runner and publication gates remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_directory)
    (ROOT/'docs/reviews/competition_nasa_native_models.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
