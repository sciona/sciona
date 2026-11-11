"""Qualify reusable helix cylinder and plane intersections for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_intersection_providers import reviewed_plan, ROOT, sha, connect
from scripts.review_trackml_intersection_reuse import review as reuse
from scripts.review_trackml_geometry_provider_semantics import packages

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Caller supplies finite aligned float64 vectors, consistent length units, and initial points on the specified helix. Pitch is axial advance per revolution. Reference surfaces and movement bounds are explicit for cross-domain reuse.',
    'Cylinder requires positive radius and target radius squared, nonzero pitch and axial direction, and nonzero transverse center magnitude. Empty batches are supported. Concentric degeneracy rejects.',
    'Cylinder considers four retained source phase candidates. A signed minimum phase permits backward reselection; no unbounded multi-revolution search is claimed. Geometric misses return NaN coordinates and phase with a false hit flag.',
    'Cylinder computes orientation from separate signs, avoiding product underflow or overflow. This intentionally differs from the source when multiplying tiny direction and pitch underflows to zero; exact source comparisons cover the tested ordinary numeric domain.',
    'Plane requires nonempty vectors and nonzero pitch. It intersects the axial plane in either direction; layer selection and empty selected batches are recipe responsibilities. Its radius argument is unused by the coordinate calculation.',
    'Plane preconditions do not enforce every qualified shape, dtype or geometric consistency condition. Callers must satisfy the recorded domain; witness output shapes do not validate those conditions.',
    'The plane symbolic expression specifies phase advance only, not a proof of full coordinate rotation. Its witness returns output shapes and dtypes only. Cylinder witness rejects abstract execution.',
    'Plane symbolic validity metadata has a zero lower bound for pitch and pitch-per-radian, while concrete execution supports either nonzero sign. This review qualifies the concrete signed-pitch interface only and does not extend symbolic validity metadata.',
    'Floating-point range and rounding remain limits. No clipping or finite replacement of cylinder misses. A reachable arccos argument outside its floating-point domain rejects rather than clipping.',
    'Detector layout, identifiers, lateral extent, correction maps, layer selection, multiplicity and recursive retries remain in the recipe; provider approval does not qualify those policies.',
    'Synthetic source parity and analytical invariants do not establish arbitrary-input accuracy, historical competition accuracy or clean deployment on other platforms.',
    'Installed RECORD validation checks current package files, not publisher authentication.',
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
        'tests/test_helix_cylinder_provider.py', 'tests/test_intersection_provider_semantics.py',
        'scripts/review_trackml_intersection_provider_semantics.py', 'scripts/review_trackml_intersection_reuse.py',
        'scripts/review_trackml_geometry_provider_semantics.py', 'scripts/stage_trackml_intersection_providers.py',
        'sciona/trackml_intersections.py', 'sciona/ghost/registry.py']}
    for relative in ['physics/helix_cylinder.py', 'particle_tracking/track_matching/atoms.py',
                     'particle_tracking/track_matching/expressions.py', 'particle_tracking/track_matching/witnesses.py',
                     'particle_tracking/source_tracking/solution_LICENSE.source']:
        key = 'sciona-atoms-physics/src/sciona/atoms/'+relative
        paths[key] = ROOT.parent/key
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    comparison = reuse()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='intersection-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_helix_cylinder_provider.py', 'tests/test_intersection_provider_semantics.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Intersection contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=132, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 132-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    for record in intake['providers']:
        cylinder = record['runtime_fqdn'].endswith('.helix_cylinder_intersection')
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'],
            operation=('Intersect a helix with a caller-selected coaxial cylinder under an explicit phase movement bound; return coordinates, signed phase and hit mask.' if cylinder else
                       'Intersect a helix with a caller-selected axial plane; return coordinates and signed phase.'),
            plain_description=('Find where a spiral path meets a chosen round surface, or report no crossing.' if cylinder else 'Find where a spiral path meets a chosen flat surface.'),
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Caller-defined helix and reference surface; no detector layout, layer selection, correction or ranking policy.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-intersection-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_source_cases=dict(cylinder_batches=96,cylinder_batch_size=32,
                                            plane_batches=comparison['exact_plane_batches'],plane_batch_size=comparison['helices_per_plane_batch']),
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_intersection_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_intersection_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
