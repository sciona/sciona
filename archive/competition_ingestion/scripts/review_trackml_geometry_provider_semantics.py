"""Qualify the existing circle and helix geometry provider for canonical Tier 3 publication."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_geometry_canonical_records import reviewed_plan,ROOT,sha,atoms
from scripts.review_trackml_geometry_reuse import review as reuse
import base64
import hashlib
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LIMITS=[
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Nonempty aligned finite float64 vectors with consistent caller-defined length units; require finite intermediate arithmetic, nonzero tangent magnitude for circle fallback and nonzero least-squares phase variance.',
    'Known finite substitutions at coincident origin and zero phase variance differ from retained source. Source-compatible adapters reject them; empty internal batches bypass these nonempty providers.',
    'The providers do not independently validate every stated shape, unit or provenance precondition. Qualified callers must enforce them; synthetic checks are not proof over all rounded inputs.',
    'Two-point and least-squares phase conventions differ at half-turn boundaries. Ordered successive points under half a turn are the regular qualified domain; explicit zero-pitch replacement remains caller policy.',
    'Geometry is reusable across applications with consistent units. Detector topology, candidate ranking, magnetic-field interpretation, calibration maps and tracking lifecycle remain separate.',
    'Installed RECORD validation does not authenticate publisher wheels, clean deployment or other platforms.',
    'Legacy history and approval remain preserved. Canonical publication and original TrackML graph qualification remain separate actions.',
]


def packages():
    pending = ['numpy','sympy','icontract','pydantic','threadpoolctl']
    result = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in result: continue
        dist = metadata.distribution(name); requirements = []
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({'extra': ''}): continue
            if req.extras or req.url: raise ValueError('Unreviewed extra or URL requirement')
            if not req.specifier.contains(metadata.version(req.name), prereleases=True):
                raise ValueError('Unsatisfied runtime requirement: '+str(req))
            requirements.append(str(req)); pending.append(req.name)
        verified = 0
        for relative in dist.files or []:
            if relative.hash is None: continue
            path = dist.locate_file(relative)
            with path.open('rb') as stream: digest = hashlib.file_digest(stream, relative.hash.mode).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b'=').decode() != relative.hash.value:
                raise ValueError('Installed RECORD mismatch: '+name)
            verified += 1
        if not verified: raise ValueError('No hashed files: '+name)
        result[name] = dict(version=dist.version, hashed_files_verified=verified, active_requirements=sorted(requirements))
    return dict(sorted(result.items()))



def review():
    intake=reviewed_plan()
    paths={p:ROOT/p for p in ('tests/test_trackml_geometry_provider_reuse.py',
        'scripts/review_trackml_geometry_provider_semantics.py','scripts/review_trackml_geometry_reuse.py',
        'scripts/stage_trackml_geometry_canonical_records.py','scripts/audit_trackml_geometry_intake.py',
        'sciona/trackml_geometry.py','sciona/trackml_helix.py','sciona/ghost/decorators.py','sciona/ghost/registry.py','sciona/ghost/dimensions.py')}
    for name in ('atoms.py','expressions.py','witnesses.py','__init__.py'):
        key='sciona-atoms-physics/src/sciona/atoms/particle_tracking/helix_geometry/'+name
        paths[key]=ROOT.parent/key
    before={k:sha(p) for k,p in paths.items()};installed=packages();comparison=reuse()
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='trackml_geometry-synthetic-') as directory:
        xml=Path(directory)/'tests.xml'
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q','tests/test_trackml_geometry_provider_reuse.py','--junitxml='+str(xml)],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        if run.returncode:raise ValueError('Independent geometry tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=15,failures=0,errors=0,skipped=0):raise ValueError('Complete 15-test suite required')
    if before!={k:sha(p) for k,p in paths.items()} or installed!=packages():raise ValueError('Qualified software changed')
    if intake!=reviewed_plan():raise ValueError('Qualified intake changed')
    providers=[]
    descriptions={
        'circle_from_three_points':('Fit a circle through each numeric point triple with an explicit radius cap.','Fit circles through groups of three points.'),
        'helix_pitch_from_two_points':('Estimate axial advance per revolution from two points and an explicit center.','Estimate how far a spiral rises during one turn.'),
        'helix_pitch_least_squares':('Fit axial advance per revolution by least squares over three ordered points and an explicit center.','Fit a spiral rise using three points and report its squared error.')}
    for record in intake['records']:
        name=record['artifact']['fqdn'].rsplit('.',1)[-1]
        operation,plain=descriptions[name]
        providers.append(dict(artifact_id=record['artifact']['artifact_id'],fqdn=record['artifact']['fqdn'],
            version_id=record['version']['version_id'],content_hash=record['version']['content_hash'],interfaces=record['interfaces'],
            operation=operation,plain_description=plain,validity_bounds=' '.join(p['constraints'] for p in record['interfaces']),
            reuse='Numeric geometry with caller-defined consistent units; no detector topology, field strength, particle identity or calibration map is hardcoded.',
            registered_provider_dependencies=[]))
    return dict(format='trackml_geometry-provider-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=providers,tests=counts,
        exact_source_cases=comparison['exact_regular_cases'],singular_differences=comparison['singular_differences'],empty_input_restrictions=comparison['empty_input_restrictions'],software_sha256=before,installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_geometry_canonical_intake_plan.json'),
        resource_policy=dict(serial_tests=True,native_threads=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/trackml_geometry_provider_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=result['tests'],providers=len(result['providers']),approved=False)))
