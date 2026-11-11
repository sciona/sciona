"""Qualify reusable dense Lloyd clustering for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_kmeans_providers import reviewed_plan, ROOT, sha, connect
import base64
import hashlib
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Nonempty finite float32 or float64 points with caller-defined Euclidean feature scaling; input precision preserved and storage copied.',
    'K must be a positive integer no greater than N. Duplicate points can yield fewer than K distinct clusters; backend warnings remain visible.',
    'Unsigned 32-bit integer seed and positive restart/iteration counts required. Initialization uses k-means++; updates use Lloyd. Finite nonnegative tolerance has installed backend semantics.',
    'Iteration limits and convergence tolerance do not promise global optimality or converged centers. Labels preserve caller row order and index returned centers; cluster numbering is backend-dependent.',
    'Finite inputs may overflow intermediate arithmetic. Nonfinite returned centers reject; exhaustive floating-point accuracy is not claimed.',
    'Abstract witness rejects; no symbolic proof of iterative clustering is claimed.',
    'One native thread per fit. Caller owns aggregate concurrency. No historical accuracy or cross-version reproducibility claim.',
    'No ring count, detector layout, grouping, identifier mapping or gap detection is embedded; these remain recipe policy.',
    'Installed RECORD checks validate current dependency files, not publisher authentication or clean deployment on other platforms.',
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
        'tests/test_lloyd_cluster_provider.py', 'scripts/review_trackml_kmeans_provider_semantics.py',
        'scripts/stage_trackml_kmeans_providers.py', 'sciona/ghost/registry.py',
        'sciona/decomposed_trackml_detector_lifecycle.py']}
    relative = 'sciona-atoms-ml/src/sciona/atoms/ml/sklearn/cluster/lloyd_fit.py'
    paths[relative] = ROOT.parent/relative
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='kmeans-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_lloyd_cluster_provider.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Clustering contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=67, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 67-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    descriptions = {
        'fit_lloyd_clusters': ('Fit dense Euclidean clusters using k-means++ initialization and Lloyd updates with explicit seed, restart, iteration and tolerance controls.', 'Group nearby points and return each group center and each point assignment.'),
    }
    for record in intake['providers']:
        operation, plain = descriptions[record['runtime_fqdn'].rsplit('.', 1)[-1]]
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'], operation=operation, plain_description=plain,
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Generic Euclidean clustering over caller-supplied features; feature scaling, detector grouping, ring counts and gap detection remain recipe policy.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-kmeans-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_backend_cases=36,
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_kmeans_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_kmeans_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
