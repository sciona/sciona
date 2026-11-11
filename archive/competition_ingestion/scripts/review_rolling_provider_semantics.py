"""Qualify existing generic rolling providers for automated Tier 3 review."""
import base64
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from sciona.atoms.ml.tabular.gradient_boosting import atoms, witnesses
from sciona.ghost import registry, abstract, dimensions
from scripts.stage_rolling_canonical_records import reviewed_plan, ROOT, sha

TESTS = ['tests/test_rolling_provider_reuse.py','tests/test_decomposed_future_sales_features.py']
LIMITS = [
    'Automated Tier 3 qualification only; no Tier 1 human review or Tier 2 usage qualification.',
    'Reusable numerical trailing statistics include the current row and partial startup windows; forecasting callers must shift and mask as appropriate. The Future Sales graph binds this exact provider and applies both policies explicitly.',
    'Supported publication inputs are finite nonempty float64 vectors, positive Python integer widths and nonempty ordered lists of mean/std/min/max/sum. The implementation also accepts empty vectors, boolean widths and positive float widths converted with int; strict supported types are caller responsibilities.',
    'std uses population variance (ddof=0). NumPy floating-point reduction and overflow behavior remain visible; nonfinite aggregate outputs fail the registered postcondition.',
    'Witness metadata does not prove concrete numerical execution or statistical validity.',
    'Installed files match their own RECORD hashes; independent publisher authentication, optional extras, clean deployment and other platforms are not certified.',
    'Synthetic numerical correctness and pipeline parity do not establish empirical forecast accuracy or generalization.',
    'This review does not approve canonical publication or the corrected original Future Sales CDG.',
]


def packages():
    pending = ['numpy', 'scipy', 'icontract', 'pydantic', 'lightgbm', 'optuna']
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
    intake = reviewed_plan()
    paths = {p: ROOT/p for p in TESTS + ['scripts/review_rolling_provider_semantics.py',
        'scripts/stage_rolling_canonical_records.py','scripts/audit_rolling_provider_intake.py',
        'sciona/decomposed_future_sales_features.py','sciona/future_sales_features.py',
        'sciona/future_sales_training.py','sciona/future_sales_worker.py','sciona/future_sales_contract.py',
        'scripts/future_sales_synthetic.py']}
    for module in (atoms,witnesses,registry,abstract,dimensions):
        paths[module.__name__]=Path(inspect.getfile(module))
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='rolling-synthetic-review-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS, '--junitxml='+str(xml)],
            cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode: raise ValueError('Provider tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {key: sum(int(s.get(key, '0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    if counts != dict(tests=63, failures=0, errors=0, skipped=0): raise ValueError('Complete 63-test suite required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages():
        raise ValueError('Software changed during qualification')
    if intake != reviewed_plan(): raise ValueError('Intake changed during qualification')
    records = []
    for item in intake['records']:
        records.append(dict(artifact_id=item['artifact']['artifact_id'],fqdn=item['artifact']['fqdn'],
            version_id=item['version']['version_id'],content_hash=item['version']['content_hash'],interfaces=item['interfaces'],
            operation='Compute ordered trailing mean, population standard deviation, minimum, maximum and sum with partial startup windows including the current row.',
            plain_description='For each position, summarize the values in the trailing window, including that position. Use the available shorter window at the beginning.',
            validity_bounds=' '.join(p['constraints'] for p in item['interfaces'] if p['direction']=='input'),
            reuse='Domain-independent numerical sequence statistics; no retail, period-panel, entity or competition-specific input required.',
            registered_provider_dependencies=[]))
    return dict(format='rolling-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True,
        providers=records, tests=counts, software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/rolling_canonical_intake_plan.json'),
        resource_policy=dict(native_threads=1, test_workers=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/rolling_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=result['tests'],
        packages=len(result['installed_packages']), approved=False)))
