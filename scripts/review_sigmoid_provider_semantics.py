"""Qualify existing generic calibration providers for automated Tier 3 review."""
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
from sciona.atoms.ml.sklearn.calibration import atoms, state_models, witnesses
from sciona.ghost import registry, abstract, dimensions
from sciona.atoms.numerical.statistics import ordered_sum
from scripts.stage_sigmoid_canonical_records import reviewed_plan, ROOT, sha

TESTS = ['tests/test_sigmoid_provider_reuse.py', 'tests/test_decomposed_march_runtime.py',
         'tests/test_march_features.py', 'tests/test_march_training.py', 'tests/test_march_contract.py']
LIMITS = [
    'Automated Tier 3 qualification only; no human review or Tier 2 usage certification.',
    'Synthetic mechanical reuse and reference parity do not establish empirical calibration, accuracy or generalization in any domain.',
    'The supported catalog interface uses finite float64 binary scores and 0/1 labels; independence and score meaning are caller responsibilities, not authenticated by the provider.',
    'Runtime decorators check shape and finite output state but do not enforce every documented dtype, finite-input, class-balance or weight restriction. Callers must satisfy the full supported input contract.',
    'The installed private sklearn sigmoid fitter returns optimizer coefficients without checking solver success; finite coefficients do not certify convergence or optimality.',
    'The registered witness is symbolic metadata, not evidence that concrete fitting succeeded.',
    'Installed files match their own RECORD hashes; independent publisher authentication, optional extras, clean deployment and other platforms are not certified.',
    'This review does not publish the providers or certify execution/publication of the corrected March CDG.',
]


def packages():
    pending = ['numpy', 'scipy', 'scikit-learn', 'icontract', 'networkx', 'pydantic']
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
    paths = {p: ROOT/p for p in TESTS + ['scripts/review_sigmoid_provider_semantics.py',
        'scripts/stage_sigmoid_canonical_records.py', 'scripts/audit_sigmoid_provider_intake.py',
        'sciona/decomposed_march_runtime.py', 'sciona/march_contract.py', 'sciona/march_training.py',
        'sciona/march_features.py', 'scripts/march_synthetic.py']}
    for module in (atoms, state_models, witnesses, registry, abstract, dimensions, ordered_sum):
        paths[module.__name__] = Path(inspect.getfile(module))
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='sigmoid-synthetic-review-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS, '--junitxml='+str(xml)],
            cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode: raise ValueError('Provider tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {key: sum(int(s.get(key, '0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    if counts != dict(tests=88, failures=0, errors=0, skipped=0): raise ValueError('Complete 88-test suite required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages():
        raise ValueError('Software changed during qualification')
    if intake != reviewed_plan(): raise ValueError('Intake changed during qualification')
    records = []
    for item in intake['records']:
        fit = item['artifact']['fqdn'].endswith('_fit')
        records.append(dict(artifact_id=item['artifact']['artifact_id'], fqdn=item['artifact']['fqdn'],
            version_id=item['version']['version_id'], content_hash=item['version']['content_hash'],
            interfaces=item['interfaces'],
            operation=('Fit Platt sigmoid slope and intercept using the installed sklearn binary calibration routine, with optional sample weights.'
                       if fit else 'Apply a fitted sigmoid to raw scores using expit(-(a*score+b)).'),
            plain_description=('Learn two numbers that convert model scores into probabilities, using separate labeled examples and optional weights.'
                               if fit else 'Convert model scores into probabilities using two previously fitted numbers.'),
            validity_bounds=' '.join(p['constraints'] for p in item['interfaces'] if p['direction']=='input'),
            reuse='Domain-independent numeric calibration; no season, sport, team, task-specific feature layout or original competition data required.',
            registered_provider_dependencies=[]))
    return dict(format='sigmoid-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True,
        providers=records, tests=counts, software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/sigmoid_canonical_intake_plan.json'),
        resource_policy=dict(native_threads=1, test_workers=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/sigmoid_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=result['tests'],
        packages=len(result['installed_packages']), approved=False)))
