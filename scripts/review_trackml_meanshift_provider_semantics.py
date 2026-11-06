"""Qualify reusable MeanShift clustering for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_meanshift_providers import reviewed_plan, ROOT, sha, connect
import base64
import hashlib
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from scripts.review_trackml_kmeans_provider_semantics import packages
from scripts.stage_trackml_meanshift_providers import legacy_digest, LEGACY_IDS

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Nonempty finite dense numeric points and optional seeds are converted to float64; float32 backend execution is not preserved. Caller owns Euclidean feature scaling.',
    'Positive finite explicit bandwidth or None for bandwidth estimation. Estimates can be zero on degenerate input; universal nondegenerate estimation is not promised.',
    'Seeds may be caller supplied, input rows, or populated grid bins. Empty seeds or no populated bins are not qualified; seeds with no neighbors reject.',
    'min_bin_freq is a positive integer. max_iter is nonnegative; zero still performs the source first mean update and does not mean no computation.',
    'cluster_all=false permits -1 labels. Centers and label numbering retain installed backend ordering, not a stable cross-version identity.',
    'Finite inputs may overflow intermediate arithmetic. No exhaustive floating-point accuracy, global optimum or historical competition accuracy claim.',
    'Qualification uses n_jobs=1 with native threads externally bounded to one; callers must bound total concurrency.',
    'Existing witness and symbolic metadata are not a proof of iterative clustering semantics.',
    'Ring counts, detector topology, gap detection and grouping remain separate recipe policy.',
    'Installed RECORD checks validate current dependency files, not publisher authentication or deployment on other platforms.',
    'Legacy approval and serving remain preserved; canonical publication and original CDG qualification are separate actions.',
]


def staged(intake):
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if legacy_digest(db, LEGACY_IDS) != intake['legacy_sha256']: raise ValueError('Legacy history changed')
        for record in intake['providers']:
            identity, version = record['atom_id'], record['version_id']
            for prefix, key, status in [('atom', 'atom_id', 'approved'), ('artifact', 'artifact_id', 'draft')]:
                row = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.version_id=%s''',
                    (identity, version)).fetchone()
                if row != dict(fqdn=record['catalog_fqdn'], status=status, is_publishable=(prefix=='atom'),
                               content_hash=record['content_hash'], trust_tier=3, is_latest=True):
                    raise ValueError('Exact staged provider required')
                ports = db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
                if ports != ([] if prefix=='atom' else record['interfaces']):
                    raise ValueError('Exact staged interface required')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (version,)).fetchone():
                raise ValueError('Unexpected registered provider dependency')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('failed','completed')", (version,)).fetchone():
                raise ValueError('Completed failed audit exists')


def review():
    intake = reviewed_plan()
    staged(intake)
    paths = {p: ROOT/p for p in [
        'tests/test_meanshift_provider_semantics.py', 'scripts/review_trackml_meanshift_provider_semantics.py',
        'scripts/stage_trackml_meanshift_providers.py', 'sciona/ghost/registry.py',
        'sciona/decomposed_trackml_detector_lifecycle.py']}
    relative = 'sciona-atoms-ml/src/sciona/atoms/ml/sklearn/cluster/atoms.py'
    paths[relative] = ROOT.parent/relative
    paths['scripts/review_trackml_kmeans_provider_semantics.py'] = ROOT/'scripts/review_trackml_kmeans_provider_semantics.py'
    for name in ('state_models.py', 'witnesses.py'):
        relative = 'sciona-atoms-ml/src/sciona/atoms/ml/sklearn/cluster/'+name
        paths[relative] = ROOT.parent/relative
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='meanshift-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_meanshift_provider_semantics.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Clustering contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=28, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 28-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    descriptions = {
        'mean_shift': ('Find Euclidean density modes by repeated neighborhood means, suppress nearby modes, and assign points to retained centers.', 'Find groups of nearby points and report their centers and assignments.'),
    }
    for record in intake['providers']:
        operation, plain = descriptions[record['runtime_fqdn'].rsplit('.', 1)[-1]]
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'], operation=operation, plain_description=plain,
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Generic Euclidean mode clustering over caller-supplied features; detector topology and grouping stay outside the operation.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-meanshift-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_backend_cases=26,
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_meanshift_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_meanshift_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
