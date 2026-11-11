"""Full 35-fit/98,000-round component check on generated numerical inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from sciona.flavours_mass_correction import correct_mass


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'sciona/flavours_mass_correction.py'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    rng = np.random.default_rng(712)
    train = rng.normal(size=(75, 6))
    query = rng.normal(size=(9, 6))
    # A constant target makes the independently known population answer exact
    # in the limit, while still exercising every actual tree-building call.
    result = correct_mass(train, np.full(75, 2.0), query)
    assert len(result['fit_rounds']) == 35
    assert set(result['fit_rounds']) == {2800}
    assert np.all(result['oof_coverage'] == 7)
    np.testing.assert_allclose(result['oof_mass'], 2.0, atol=2e-5, rtol=0)
    np.testing.assert_allclose(result['query_mass'], 2.0, atol=2e-5, rtol=0)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    report = dict(passed=True, synthetic_only=True, catalog_mutations=0,
                  component_only=True, model_fits=35, boosting_rounds=98000,
                  per_row_oof_coverage=7, constant_target_oracle=True,
                  source_sha256=before, xgboost_version=result['runtime_version'],
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Full regression budgets; no full pipeline or competition accuracy claim. '
                        'Modern exact-tree CPU XGBoost with fixed 0.5 base score; '
                        'historical implementation parity not established.')
    (root / 'docs/reviews/competition_flavours_mass_correction_validation.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
