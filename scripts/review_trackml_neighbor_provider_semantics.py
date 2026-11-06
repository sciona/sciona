"""Qualify reusable fitted Euclidean search for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_neighbor_providers import reviewed_plan, ROOT, sha, connect
import base64
import hashlib
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Nonempty finite float64 matrices with caller-defined consistent feature scaling. Query dimension must match fitted dimension. No detector layout, labels, layer partitioning, projections or identifier mapping is embedded.',
    'Fit copies the input points and returns an opaque runtime scikit-learn state. Callers must not modify the state; validation of its public configuration is not a complete integrity check of estimator internals. No portable state serialization is qualified.',
    'Fit default neighbor count influences auto backend selection even if queries supply a different k. Leaf size can affect backend order. Defaults are 5 and 30; both controls require positive integers.',
    'Explicit self queries include self. k must not exceed fitted row count. Radius is finite, nonnegative and inclusive, with zero and empty match rows supported. Empty query batches reject.',
    'Tied k-neighbor selection and unsorted radius ordering follow the installed backend. No cross-version or cross-platform ordering guarantee. Existing stable dense search is intentionally not substituted.',
    'Finite inputs can overflow intermediate distance arithmetic. Nonfinite returned distances reject; exhaustive accuracy over all floating-point magnitudes is not claimed.',
    'Abstract witnesses reject: concrete fitted state is required. No symbolic proof is claimed.',
    'Each operation uses one worker and one native thread. Caller owns aggregate concurrency and state lifetime.',
    'Synthetic exact-backend comparisons and lifecycle agreement establish the tested behavior, not historical competition accuracy or accuracy for arbitrary new applications.',
    'Installed RECORD validation checks current dependency files and active requirements, not publisher authentication or clean deployment on another platform.',
    'Provider publication and original TrackML CDG qualification remain separate actions.',
]


def packages():
    pending = ['numpy','sympy','icontract','pydantic','threadpoolctl','scikit-learn']
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
        'tests/test_fitted_neighbor_search.py', 'scripts/review_trackml_neighbor_provider_semantics.py',
        'scripts/stage_trackml_neighbor_providers.py', 'sciona/ghost/registry.py',
        'sciona/decomposed_trackml_neighbor_lifecycle.py']}
    relative = 'sciona-atoms-ml/src/sciona/atoms/ml/sklearn/neighbors/fitted_search.py'
    paths[relative] = ROOT.parent/relative
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='neighbor-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_fitted_neighbor_search.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Neighbor contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=66, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 66-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    descriptions = {
        'fit_euclidean_search': ('Fit an auto-selected Euclidean neighbor index with explicit default neighbor count and leaf size.', 'Build a reusable index for finding nearby points.'),
        'query_euclidean_k': ('Query k nearest fitted rows, preserving backend tie selection and distance order.', 'Find a chosen number of nearby points and their distances.'),
        'query_euclidean_radius': ('Query all fitted rows within an inclusive distance radius, preserving unsorted backend order.', 'Find points within a chosen distance, including when none match.'),
    }
    for record in intake['providers']:
        operation, plain = descriptions[record['runtime_fqdn'].rsplit('.', 1)[-1]]
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'], operation=operation, plain_description=plain,
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Generic Euclidean search over caller-supplied numeric features; domain grouping, coordinate transforms and labels remain outside these operations.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-neighbor-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_backend_cases=48,
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_neighbor_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_neighbor_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
