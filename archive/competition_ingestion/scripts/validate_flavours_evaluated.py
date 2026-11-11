"""Execute the complete training/prediction composition on synthetic inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sciona.flavours_evaluated_execution import execute_evaluated


def main():
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / 'sciona').glob('flavours_*.py'))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    rng = np.random.default_rng(91)
    def population(n):
        base = rng.normal(size=(n,46))
        label = np.arange(n) % 2
        base[:, :8] = (label * 2 - 1)[:, None]
        return (base, np.full((n,3),5.), np.full((n,3),3.), np.full(n,5.),
                np.full(n,2.), np.ones(n)), label
    train, labels = population(600)
    query, query_labels = population(20)
    print('START complete Flavours training/prediction pipeline', flush=True)
    agreement, _ = population(40)
    correlation, _ = population(400)
    controls = dict(agreement_a=agreement, agreement_b=agreement,
                    correlation=correlation, weights_a=np.ones(40), weights_b=np.ones(40),
                    correlation_mass=rng.normal(size=400), quality=np.ones(600))
    evaluated = execute_evaluated(train, labels, np.full(600,6.5), query, controls, 7)
    result = evaluated['execution']
    assert evaluated['evaluation']['ks_agreement'] == 0
    assert evaluated['evaluation']['agreement_passed']
    assert np.isfinite(evaluated['evaluation']['cvm_mass'])
    assert np.isfinite(evaluated['evaluation']['truncated_weighted_auc'])
    assert len(result['mass_correction']['fit_rounds']) == 35
    assert sum(result['mass_correction']['fit_rounds']) == 98000
    assert len(result['classifiers']['fits']) == 30
    assert sum(f['rounds'] for f in result['classifiers']['fits']) == 13458
    assert len(result['neural']['fits']) == 50
    assert sum(f['epochs'] for f in result['neural']['fits']) == 150000
    assert sum(f['optimizer_updates'] for f in result['neural']['fits']) == 600000
    assert result['neural']['checkpoint_replays_exact']
    scores = evaluated['query_scores']
    assert scores.shape == (20,) and np.isfinite(scores).all()
    assert scores[query_labels == 1].mean() > scores[query_labels == 0].mean()
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(passed=True, synthetic_only=True, catalog_mutations=0,
                  model_fits=115, tree_boosting_rounds=111458, neural_epochs=150000,
                  neural_optimizer_updates=600000, source_sha256=hashes,
                  neural_checkpoint_replays_exact=True, evaluation=evaluated['evaluation'],
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Integrated physical features, mass regression, five classifiers, '
                        'neural ensemble, final nonlinear scores and all source metrics. '
                        'Identical agreement inputs intentionally test the zero-KS oracle. Catalog graph execution pending; '
                        'no historical execution or competition accuracy claim.')
    (root / 'docs/reviews/competition_flavours_evaluated_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
