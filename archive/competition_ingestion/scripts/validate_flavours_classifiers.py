"""All five classifiers, five folds and full refits, with synthetic inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from sciona.flavours_classifiers import train_classifiers


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'sciona/flavours_classifiers.py'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    rng = np.random.default_rng(45)
    x = rng.normal(size=(200, 8))
    q = rng.normal(size=(20, 8))
    y = (x[:, 0] > 0).astype(float)
    def views(a):
        return {'base': a, 'restricted': a[:, :4],
                'proxy': np.column_stack([a, a[:, 0] ** 2]),
                'corrected': np.column_stack([a, a[:, :4] ** 2])}
    result = train_classifiers(views(x), y, views(q))
    expected = {'xgb1': 5, 'xgb2': 35, 'xgb3': 700, 'xgb4': 3, 'xgb5': 1500}
    assert len(result['fits']) == 30
    for branch, rounds in expected.items():
        fits = [f for f in result['fits'] if f['branch'] == branch]
        assert len(fits) == 6 and {f['fold'] for f in fits} == {0, 1, 2, 3, 4, 'full'}
        assert all(f['rounds'] == rounds for f in fits)
    # Branch3 sees a simple binary separator; this checks actual signal learning.
    accuracy = float(np.mean((result['query']['xgb3'] >= .5) == (q[:, 0] > 0)))
    assert accuracy >= .9
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    report = dict(passed=True, synthetic_only=True, component_only=True,
                  catalog_mutations=0, model_fits=30, boosting_rounds=13458,
                  checkpoint_replays_exact=result['checkpoint_replays_exact'],
                  synthetic_separator_accuracy=accuracy,
                  source_sha256=before, xgboost_version=result['runtime_version'],
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Full classifier budgets with synthetic prepared views. '
                        'No integrated feature-generation, neural branch, competition accuracy '
                        'or historical runtime parity claim.')
    (root / 'docs/reviews/competition_flavours_classifiers_validation.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
