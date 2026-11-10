"""Validate reusable whole-group distribution scoring with synthetic populations."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from sciona.atoms.ml import group_distribution_matching as provider
from sciona.atoms.ml.constrained_ml.decorrelation import atoms as ks_provider
from scripts.import_flavours_provider_drafts import connect, ROOT
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empirical_distance(a, b):
    support = np.unique(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), support, side='right')/len(a)
    fb = np.searchsorted(np.sort(b), support, side='right')/len(b)
    return np.max(np.abs(fa-fb))


def validate():
    paths = [Path(inspect.getfile(provider)), Path(inspect.getfile(ks_provider)),
             ROOT.parent/'sciona-atoms-ml/tests/test_group_distribution_matching_provider.py', Path(__file__)]
    before = {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows = db.execute("""SELECT a.artifact_id::text,a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.fqdn=%s AND v.is_latest AND a.status='approved' AND a.is_publishable
            AND artifact_is_publishable(a.artifact_id)
            AND EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id)""",
            ('sciona.atoms.ml.constrained_ml.decorrelation.compute_ks_agreement',)).fetchall()
        if len(rows) != 1:
            raise ValueError('Unique approved served KS dependency required')
    repo = ROOT.parent/'sciona-atoms-ml'
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.fqdn == rows[0]['fqdn']]
    if (len(specs) != 1 or str(specs[0].version_id) != rows[0]['version_id']
            or specs[0].content_hash != rows[0]['content_hash']):
        raise ValueError('Runtime KS implementation differs from approved catalog version')
    rng = np.random.default_rng(463)
    cases = []
    for domain, candidate_count in [('synthetic-instrument-batches', 10000), ('synthetic-production-batches', 64)]:
        labels = list(range(17))
        if domain == 'synthetic-instrument-batches':
            groups = np.repeat(labels, 8)
            training = rng.normal(size=(len(groups), 4)) + groups[:, None]/10
            reference = rng.normal(.8, 1.2, size=(96, 4))
            construction = 'Continuous synthetic measurements with batch-dependent location shifts and equal batch sizes.'
        else:
            groups = np.repeat(labels, np.arange(1, 18))
            rates = 1 + groups[:, None]/4 + np.arange(4)[None, :]
            training = rng.poisson(rates).astype(np.float64)
            reference = rng.poisson([2., 4., 5., 7.], size=(96, 4)).astype(np.float64)
            construction = 'Discrete synthetic counts with ties, unequal batch sizes and feature-specific rates.'
        candidates = np.zeros((candidate_count, len(labels)), dtype=bool)
        for row in candidates:
            row[rng.choice(len(labels), 10, replace=False)] = True
        actual = provider.group_subset_ks(training, reference, groups, candidates,
                                         labels, ['f0', 'f1', 'f2', 'f3'], domain+'.features.v1')
        expected = np.empty_like(actual)
        for i, mask in enumerate(candidates):
            subset = training[np.isin(groups, np.array(labels)[mask])]
            for j in range(4):
                expected[i, j] = empirical_distance(subset[:, j], reference[:, j])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        # Averaging is an explicit application policy, outside the reusable provider.
        scores, oracle_scores = actual.mean(axis=1), expected.mean(axis=1)
        np.testing.assert_allclose(scores, oracle_scores, rtol=0, atol=1e-12)
        best = int(np.argmin(scores))
        if oracle_scores[best] > oracle_scores.min()+1e-12:
            raise ValueError('Selected subset is not an oracle minimizer')
        cases.append(dict(synthetic_domain=domain, construction=construction, candidates=candidate_count, groups=17,
            selected_groups_per_candidate=10, feature_count=4, distances_checked=int(actual.size),
            maximum_absolute_error=float(np.max(np.abs(actual-expected))), selected_oracle_minimizer=True))
    if before != {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}:
        raise ValueError('Implementation changed during qualification')
    return dict(format='lanl-group-distribution-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, reused_catalog_dependencies=rows,
        cases=cases, software_sha256=before, resource_policy=dict(concurrent_workers=1, native_threads=1),
        source='https://medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquake-prediction-competition-15a1137c2457',
        source_scope='The author describes repeated whole-group subset comparison using averaged marginal KS distances. Candidate generation here is explicit synthetic policy.',
        limitations=['Only distribution scoring and an explicit mean-distance selection example are qualified; no full LANL graph approval.',
            'Synthetic groups, candidate draws and features do not reproduce historical data or randomness.',
            'Marginal agreement does not imply joint distribution agreement or predictive accuracy; reference-population availability is a caller obligation.',
            'The author publications disagree on the feature-screening p-value inequality. This scorer computes no p-values and makes no screening decision.',
            'KS equivalence is within floating-point tolerance. Near ties require an explicit selection tolerance and tie policy in the caller.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/lanl_group_distribution_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, distances=sum(r['distances_checked'] for r in report['cases']), approved=False)))
