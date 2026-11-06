"""Bounded numeric reuse check; covariance equality is not GP lifecycle parity."""
import inspect
import json
import numpy as np
from george import kernels
from sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms import matern_kernel_matrix
from scripts.audit_decomposed_plasticc_readiness import ROOT, sha
from scripts.import_flavours_provider_drafts import connect


def review():
    module = 'sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms'
    symbol = 'matern_kernel_matrix'
    source = inspect.getfile(inspect.unwrap(matern_kernel_matrix))
    before = sha(source)
    cases = []
    for seed in (3, 17, 51):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(9, 2))*[30, 7000]
        y = rng.normal(size=(5, 2))*[30, 7000]
        for length in (1., 20., 1000.):
            scale = np.array([length, 6000.])
            for amplitude in (.25, 49.):
                reference = amplitude*kernels.Matern32Kernel(scale**2, ndim=2)
                for cross in (False, True):
                    target = y if cross else None
                    expected = reference.get_value(x, target) if cross else reference.get_value(x)
                    actual = amplitude*matern_kernel_matrix(x, target, length_scale=scale, nu=1.5)
                    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
                    cases.append(dict(seed=seed, time_length=length, amplitude=amplitude,
                                      cross_covariance=cross, max_absolute_error=float(np.max(np.abs(actual-expected)))))
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        legacy = db.execute('''SELECT a.atom_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier
            FROM atoms a LEFT JOIN atom_versions v ON a.atom_id=v.atom_id AND v.is_latest
            WHERE a.import_module=%s AND a.source_symbol=%s''', (module, symbol)).fetchall()
        canonical = []
        for record in legacy:
            canonical.extend(db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
                v.version_id::text,v.content_hash,v.trust_tier
                FROM artifacts a LEFT JOIN artifact_versions v ON a.artifact_id=v.artifact_id AND v.is_latest
                WHERE a.fqdn=%s''', (record['fqdn'],)).fetchall())
    if before != sha(source):
        raise ValueError('Candidate implementation changed')
    return dict(format='plasticc-matern-reuse.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, candidate_runtime=module+'.'+symbol, provider_sha256=before,
        legacy_catalog=legacy, canonical_catalog=canonical, cases=cases,
        observations_implementation_sha256=sha(ROOT/'sciona/plasticc_observations.py'),
        reviewer_sha256=sha(__file__),
        conclusion='Existing generic anisotropic Matern kernel matches the retained george covariance after explicit amplitude scaling with nu=1.5 and squared-to-unsquared length conversion.',
        reuse='Candidate accepts dimensionless numeric coordinate matrices and explicit length scales; astronomy band lookup, redshift policy and labels do not enter this operation.',
        limits=['Candidate qualification only; no registration or approval mutation and no new provider added.',
                'Covariance parity does not prove optimizer gradients, bounded fitting, initial-guess fallback, cached state or predictive-variance semantics.',
                'A source-faithful GP lifecycle still needs explicit amplitude policy, observation noise, fit controls and matching-state provenance.',
                'Catalog presence alone does not establish native publication eligibility or complete interface qualification.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/plasticc_matern_reuse.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, cases=len(result['cases']), legacy_records=len(result['legacy_catalog']),
                         canonical_records=len(result['canonical_catalog']), approved=False)))
