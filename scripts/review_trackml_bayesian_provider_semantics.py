"""Qualify reusable normalized residual decisions for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_bayesian_providers import reviewed_plan, ROOT, sha, connect
from scripts.review_trackml_bayesian_reuse import review as reuse
from scripts.review_trackml_geometry_provider_semantics import packages

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Caller supplies nonempty aligned finite float64 vectors. Each residual, background and positive error triple must share units; the two components can have different units. Normalized ratios and thresholds are dimensionless.',
    'Both thresholds must be finite positive scalars. Background zero is allowed. Qualified arithmetic requires finite normalized distances, cut and trust-scaled cut; overflow and singular underflow are excluded.',
    'Existing preconditions do not enforce every shape, dtype, finiteness or domain restriction. Caller boundaries must enforce the qualified domain.',
    'Good and dubious flags use independent strict comparisons and may both be true. Equality does not pass the respective comparison. These are threshold decisions, not calibrated probabilities.',
    'Provider defaults are cut_factor=3.0 and dist_trust=0.5. The source recipe uses 1.0 and 0.8. Pass the intended recipe controls explicitly.',
    'Existing symbolic metadata assigns angular dimensions and only describes normalized distance. Concrete matching-unit use is qualified; no full symbolic decision proof or generalized dimensional metadata is claimed.',
    'The witness reports boolean shapes only, without validating numeric inputs or proving decisions. Symbolic lower bounds of zero do not establish validity at zero error or thresholds.',
    'Coordinate projection, empirical error prediction, coefficient lookup and diagnostic reporting are separate recipe operations. This atom accepts numeric vectors without detector or event state.',
    'Synthetic source parity and unit-scaling checks do not establish arbitrary-input accuracy, calibrated error models or historical competition performance.',
    'Installed RECORD validation checks current package files, not publisher authentication or clean deployment on another platform.',
    'Provider publication and original TrackML CDG qualification remain separate actions.',
]


def staged(intake):
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for record in intake['providers']:
            identity, version = record['atom_id'], record['version_id']
            for prefix, key, status in [('atom', 'atom_id', 'approved' if record['existing_legacy'] else 'flagged'), ('artifact', 'artifact_id', 'draft')]:
                row = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.version_id=%s''',
                    (identity, version)).fetchone()
                if row != dict(fqdn=record['catalog_fqdn'], status=status, is_publishable=False,
                               content_hash=record['content_hash'], trust_tier=3, is_latest=True):
                    raise ValueError('Exact staged provider required')
                ports = db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
                if ports != ([] if prefix=='atom' and record['existing_legacy'] else record['interfaces']):
                    raise ValueError('Exact staged interface required')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (version,)).fetchone():
                raise ValueError('Unexpected registered provider dependency')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('failed','completed')", (version,)).fetchone():
                raise ValueError('Completed failed audit exists')


def review():
    intake = reviewed_plan()
    staged(intake)
    paths = {p: ROOT/p for p in [
        'tests/test_bayesian_provider_semantics.py', 'tests/test_decomposed_trackml_bayesian.py',
        'scripts/review_trackml_bayesian_provider_semantics.py', 'scripts/review_trackml_bayesian_reuse.py',
        'scripts/review_trackml_geometry_provider_semantics.py', 'scripts/stage_trackml_bayesian_providers.py',
        'sciona/trackml_bayesian.py', 'sciona/decomposed_trackml_bayesian_lifecycle.py', 'sciona/ghost/registry.py']}
    for relative in ['particle_tracking/track_matching/atoms.py',
                     'particle_tracking/track_matching/expressions.py', 'particle_tracking/track_matching/witnesses.py',
                     'particle_tracking/source_tracking/solution_LICENSE.source']:
        key = 'sciona-atoms-physics/src/sciona/atoms/'+relative
        paths[key] = ROOT.parent/key
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    comparison = reuse()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='bayesian-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_bayesian_provider_semantics.py', 'tests/test_decomposed_trackml_bayesian.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Decision contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=42, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 42-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    for record in intake['providers']:
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'],
            operation='Compare a two-component error-normalized residual norm against a logarithmic background-derived cut and a separate trust-scaled cut using strict inequalities.',
            plain_description='Decide whether paired numeric residuals pass a supplied background-relative threshold and independently flag residuals above a trust threshold.',
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Pure paired residual decisions with explicit numeric scales and dimensionless controls; components may use different units if each residual/background/error triple is consistent. No detector, event, lookup or calibration policy.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-bayesian-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_source_cases=dict(batches=comparison['exact_source_batches'],
                                            batch_size=comparison['batch_size']),
        source_comparison_sha256=comparison['implementation_sha256'],
        strict_boundary_cases=comparison['strict_boundary_cases'],
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_bayesian_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_bayesian_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
