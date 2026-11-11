"""Qualify generic cylindrical and plane projections for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_coordinate_providers import reviewed_plan, ROOT, sha, connect
from scripts.review_trackml_coordinate_reuse import review as reuse
from scripts.review_trackml_geometry_provider_semantics import packages

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Finite float64 coordinate arrays (N,3), including empty batches, with consistent caller-defined length units; units are a caller responsibility.',
    'Cylinder projection requires a positive finite reference radius and finite nonzero transverse norm. Transverse weight is dimensionless and may be negative or zero; it changes the output geometry and distance metric.',
    'Plane projection intersects the line through the origin and each point with the reference plane, returning x,y. Either direction is allowed, including negative scale factors; the ray wording in the source docstring does not impose a forward-only constraint.',
    'Plane projection requires nonzero input z; finite reference heights of either sign or zero are supported.',
    'The retained arithmetic order can overflow or underflow even where a rearranged real-number formula is defined. Undefined norms and nonfinite outputs reject; no clipping or finite substitution is used.',
    'Concrete execution only. The registered witness rejects abstract calls and provides no symbolic correctness or shape proof.',
    'No detector topology, particle identity, layer policy, nearest-neighbor threshold or calibration map is embedded. Geometry and metric parameters remain explicit for cross-domain reuse.',
    'Synthetic parity and geometric invariants do not establish arbitrary-input accuracy, historical competition accuracy or clean deployment on other platforms.',
    'Installed RECORD validation checks current package files, not publisher authentication.',
    'Provider publication and original TrackML CDG qualification remain separate actions.',
]


def staged(intake):
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for record in intake['providers']:
            identity, version = record['atom_id'], record['version_id']
            for prefix, key, status in [('atom', 'atom_id', 'flagged'), ('artifact', 'artifact_id', 'draft')]:
                row = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.version_id=%s''',
                    (identity, version)).fetchone()
                if row != dict(fqdn=record['catalog_fqdn'], status=status, is_publishable=False,
                               content_hash=record['content_hash'], trust_tier=3, is_latest=True):
                    raise ValueError('Exact staged provider required')
                ports = db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
                if ports != record['interfaces']:
                    raise ValueError('Exact staged interface required')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (version,)).fetchone():
                raise ValueError('Unexpected registered provider dependency')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('failed','completed')", (version,)).fetchone():
                raise ValueError('Completed failed audit exists')


def review():
    intake = reviewed_plan()
    staged(intake)
    paths = {p: ROOT/p for p in [
        'tests/test_coordinate_projection_providers.py', 'tests/test_coordinate_projection_contracts.py',
        'scripts/review_trackml_coordinate_provider_semantics.py', 'scripts/review_trackml_coordinate_reuse.py',
        'scripts/review_trackml_geometry_provider_semantics.py', 'scripts/stage_trackml_coordinate_providers.py',
        'sciona/trackml_neighbors.py', 'sciona/ghost/registry.py']}
    for relative in ['physics/coordinate_projection.py', 'particle_tracking/source_tracking/solution_LICENSE.source']:
        key = 'sciona-atoms-physics/src/sciona/atoms/'+relative
        paths[key] = ROOT.parent/key
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    comparison = reuse()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='coordinate-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_coordinate_projection_providers.py', 'tests/test_coordinate_projection_contracts.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Projection contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=42, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 42-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    for record in intake['providers']:
        cylinder = record['runtime_fqdn'].endswith('.cylindrical_projection')
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'],
            operation=('Project radial lines onto a reference cylinder using source-order arithmetic, then weight transverse coordinates.' if cylinder else
                       'Intersect lines through the origin with a reference axial plane and return their transverse coordinates.'),
            plain_description=('Map points to a cylinder and apply a chosen sideways scale.' if cylinder else 'Map points onto a flat surface along lines through the origin.'),
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Caller-defined reference geometry and consistent units; no detector layout, identifiers, ranking or calibration policy.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-coordinate-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_source_cases=dict(cylinder=comparison['exact_new_cylinder_provider_cases'],
                                            plane=comparison['exact_new_plane_provider_cases']),
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_coordinate_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_coordinate_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
