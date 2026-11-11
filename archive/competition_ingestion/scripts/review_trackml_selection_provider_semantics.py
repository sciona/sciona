"""Qualify reusable signed vector selection for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_selection_providers import reviewed_plan, ROOT, sha, connect
import base64
import hashlib
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Aligned finite float64 (N,D) matrices with D>=1; empty batches and zero vectors allowed.',
    'Absolute inner-product selection, not normalized angular comparison. Angular interpretation needs nonzero reference and equal nonzero candidate lengths.',
    'Strict first-versus-second comparison chooses second on ties; negative selected inner product reverses orientation, zero leaves it unchanged.',
    'Caller defines units and scaling. Pixel-count confidence and detector-specific cell reconstruction remain in the recipe.',
    'Nonfinite inner products reject before selection; floating-point rounding and near ties are not exhaustively bounded.',
    'Registered concrete witness rejects abstract execution; no symbolic proof is claimed.',
    'Qualification runs serially with native threads bounded to one; callers own aggregate concurrency.',
    'Installed RECORD checks validate current dependency files, not publisher authentication or clean deployment on other platforms.',
    'Synthetic numerical and lifecycle evidence does not establish historical competition accuracy. Original CDG publication remains separate.',
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
        'tests/test_vector_selection_provider.py', 'scripts/review_trackml_selection_provider_semantics.py',
        'scripts/stage_trackml_selection_providers.py', 'sciona/ghost/registry.py',
        'sciona/decomposed_trackml_cell_lifecycle.py', 'sciona/trackml_cells.py']}
    relative = 'sciona-atoms-physics/src/sciona/atoms/physics/vector_selection.py'
    paths[relative] = ROOT.parent/relative
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='selection-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_vector_selection_provider.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Vector selection contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=26, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 26-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    descriptions = {
        'select_by_absolute_inner_product': ('Select and orient one of two vectors using the larger absolute reference inner product, choosing the second on ties.', 'Choose between two directions by their alignment with a reference, then turn the choice toward that reference.'),
    }
    for record in intake['providers']:
        operation, plain = descriptions[record['runtime_fqdn'].rsplit('.', 1)[-1]]
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'], operation=operation, plain_description=plain,
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Generic signed vector selection in arbitrary positive dimension; caller controls units and scaling, while detector confidence policy remains in the recipe.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-selection-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_source_cases=4, independent_vector_cases=16,
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_selection_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_selection_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
