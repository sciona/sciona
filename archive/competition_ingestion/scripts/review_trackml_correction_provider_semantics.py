"""Qualify reusable local-frame corrections for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_correction_providers import reviewed_plan, ROOT, sha, connect
from scripts.review_trackml_correction_reuse import review as reuse
from scripts.review_trackml_geometry_provider_semantics import packages

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Caller supplies nonempty aligned finite float64 vectors and consistent length units. Coefficients have squared-length units; signed orientation is dimensionless. No particle-specific interpretation is required.',
    'Both operations require nonzero transverse coordinate radius. Cap requires finite positive absolute pitch-per-radian; cylinder requires positive helix radius and finite positive curvature denominator. Cylinder permits zero pitch.',
    'Existing preconditions do not enforce every shape, dtype or domain restriction. Safe substitutions at the coordinate axis or zero cap pitch are outside qualification. Caller boundaries must enforce the qualified domain.',
    'Cap symbolic expression describes only corrected x; cylinder expression describes only corrected z. Neither is a full vector proof. Witnesses report shapes and dtypes only, without validating geometric or numeric preconditions.',
    'Symbolic lower bounds of zero do not establish validity at singular denominators. This review qualifies concrete execution within the strict positive-denominator domain.',
    'Providers return new arrays. Coefficient lookup, missing coefficients, surface bounds, masks and in-place state updates belong to the calling recipe.',
    'Caller-provided coefficients do not imply learned calibration. No calibration training or detector-specific policy is bundled or qualified.',
    'Intermediate floating-point overflow, underflow to singular denominators, and nonfinite output are outside qualification. Synthetic ordinary-range checks do not establish arbitrary-input or historical competition accuracy.',
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
        'tests/test_correction_provider_semantics.py', 'tests/test_decomposed_trackml_corrections.py',
        'scripts/review_trackml_correction_provider_semantics.py', 'scripts/review_trackml_correction_reuse.py',
        'scripts/review_trackml_geometry_provider_semantics.py', 'scripts/stage_trackml_correction_providers.py',
        'sciona/trackml_corrections.py', 'sciona/decomposed_trackml_corrections.py', 'sciona/ghost/registry.py']}
    for relative in ['particle_tracking/detector_corrections/atoms.py',
                     'particle_tracking/detector_corrections/expressions.py', 'particle_tracking/detector_corrections/witnesses.py',
                     'particle_tracking/source_tracking/solution_LICENSE.source']:
        key = 'sciona-atoms-physics/src/sciona/atoms/'+relative
        paths[key] = ROOT.parent/key
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    comparison = reuse()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='correction-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_correction_provider_semantics.py', 'tests/test_decomposed_trackml_corrections.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Correction contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=47, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 47-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    for record in intake['providers']:
        cylinder = record['runtime_fqdn'].endswith('.perturbative_cylinder_correction')
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'],
            operation=('Apply caller-supplied axial and azimuthal displacement coefficients scaled by signed orientation and inverse helix curvature radius.' if cylinder else
                       'Apply caller-supplied radial and azimuthal displacement coefficients scaled by signed orientation and inverse absolute pitch per radian.'),
            plain_description=('Adjust coordinates along and around an axis using supplied correction coefficients.' if cylinder else 'Adjust planar coordinates outward and sideways using supplied correction coefficients.'),
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Pure local-frame displacement with explicit numeric coefficients and caller-defined units; no detector layout, coefficient lookup, masking or mutation policy.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-correction-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_source_cases=dict(batches=comparison['exact_source_batches'],
                                            batch_size=comparison['batch_size']),
        source_comparison_sha256=comparison['implementation_sha256'],
        excluded_source_differences=comparison['undefined_domain_differences'],
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_correction_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_correction_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
