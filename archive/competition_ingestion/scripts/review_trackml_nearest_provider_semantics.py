"""Qualify reusable fixed-iteration helix projection for canonical Tier 3 publication."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_nearest_canonical_records import reviewed_plan, ROOT, sha
from scripts.review_trackml_nearest_reuse import review as reuse
from scripts.review_trackml_geometry_provider_semantics import packages

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Fixed positive integer Newton iteration count. Arbitrary targets have no convergence or global-optimum guarantee; returned coordinates are an approximation.',
    'Finite scalar or aligned float64 helix parameters and nonempty aligned finite float64 target vectors, with consistent caller-defined length units, nonnegative radius and nonzero pitch.',
    'Initial coordinates must lie on the specified helix. Finite intermediate arithmetic and outputs are required; singular Newton steps, overflow and undefined zero-pitch geometry are excluded.',
    'The provider does not independently validate every input precondition. Qualified callers must enforce domain constraints; synthetic checks are not proof over all rounded inputs.',
    'The symbolic expression describes Euclidean distance, not global optimality. The ghost witness describes output shapes, not numerical correctness.',
    'The source-compatible tracking boundary bypasses empty batches; four-output tracking call sites do not require optional phase or arc-length outputs.',
    'Numeric geometry is reusable with caller-defined consistent units. Detector topology, field strength, calibration maps, candidate ranking and lifecycle policy remain separate.',
    'Installed RECORD validation does not authenticate publisher wheels, clean deployment or other platforms.',
    'Canonical publication and original TrackML CDG qualification remain separate actions; legacy history is preserved.',
]


def review():
    intake = reviewed_plan()
    names = ['tests/test_trackml_nearest_provider_reuse.py',
             'scripts/review_trackml_nearest_provider_semantics.py',
             'scripts/review_trackml_geometry_provider_semantics.py',
             'scripts/review_trackml_nearest_reuse.py',
             'scripts/stage_trackml_nearest_canonical_records.py',
             'scripts/audit_trackml_nearest_intake.py',
             'sciona/trackml_nearest.py', 'sciona/trackml_helix.py',
             'sciona/decomposed_trackml_nearest.py',
             'sciona/ghost/decorators.py', 'sciona/ghost/registry.py', 'sciona/ghost/dimensions.py']
    paths = {p: ROOT/p for p in names}
    for name in ('atoms.py', 'expressions.py', 'witnesses.py', '__init__.py'):
        key = 'sciona-atoms-physics/src/sciona/atoms/particle_tracking/helix_geometry/' + name
        paths[key] = ROOT.parent/key
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    comparison = reuse()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='nearest-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_trackml_nearest_provider_reuse.py', '--junitxml='+str(xml)],
            cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Independent geometry tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=13, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 13-test suite required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages():
        raise ValueError('Qualified software changed')
    if intake != reviewed_plan():
        raise ValueError('Qualified intake changed')
    record, = intake['records']
    provider = dict(artifact_id=record['artifact']['artifact_id'], fqdn=record['artifact']['fqdn'],
        version_id=record['version']['version_id'], content_hash=record['version']['content_hash'],
        interfaces=record['interfaces'],
        operation='Approximate a nearest point on each helix using a fixed number of Newton steps, then report Euclidean distance.',
        plain_description='Estimate a nearby point on a spiral and its distance from a target point.',
        validity_bounds=' '.join(p['constraints'] for p in record['interfaces']),
        reuse='Numeric geometry with caller-defined consistent units; no detector topology, field strength, particle identity or calibration map is hardcoded.',
        registered_provider_dependencies=[])
    return dict(format='trackml_nearest-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=[provider], tests=counts,
        exact_source_cases=comparison['exact_source_comparisons'], analytic_cases=comparison['analytic_on_helix_cases'],
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_nearest_canonical_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_nearest_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
