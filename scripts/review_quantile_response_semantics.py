"""Qualify portable quantile response providers for automated Tier 3 publication."""
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
from sciona.atoms.ml import quantile_response_encoding as atoms
from sciona.ghost import registry, abstract, dimensions
from scripts.import_quantile_response_drafts import reviewed_plan, ROOT, PLAN, sha
from scripts.import_flavours_provider_drafts import connect

LIMITS = [
    'Automated Tier 3 only; no human Tier 1 review or Tier 2 usage certification.',
    'Synthetic source parity and coordinate-transfer tests do not establish accuracy, calibration or generalization on a new application.',
    'Caller owns training-only quantile fitting, normalized response semantics and ordered probability-bin identity. Coordinate labels do not authenticate these properties.',
    'Repeated quantile boundaries are retained. Hard intervals are left-open/right-closed, so minimum and out-of-range targets may yield all-zero rows that cannot be decoded as probabilities.',
    'Soft encoding preserves direct exponential normalization. Underflow/overflow is rejected rather than silently replacing the source formula. Precision is in inverse squared coordinate units and must change when response scale changes.',
    'Decoding is a midpoint expectation in the same numeric coordinate; it does not infer inverse transforms or probability calibration. Nonfinite decoded outputs are rejected.',
    'Abstract witnesses check shapes, dtypes and coordinate identity, not concrete finiteness, sorted edges or probability normalization. Known physical dimensions require upstream normalization; unknown metadata is allowed.',
    'Six synthetic classifiers match saved boundaries, all training encodings and checkpoint predictions; neural models are not retrained by that replay. Source feature/model loaders remain diagnostic.',
    'Installed files match their own RECORD hashes and active dependency constraints. Independent package provenance, clean deployment and other platforms are not certified.',
    'This review qualifies only three quantile atoms, not the original competition CDG or the full historical ensemble.',
]
OPERATIONS = {
    'fit_quantile_response': ('Fit linearly interpolated training quantiles and retain repeated boundaries in portable state.', 'Learn value ranges from training examples for later use.'),
    'encode_quantile_response': ('Encode targets using fitted quantile midpoints with explicit hard or soft policy and inverse-square precision units.', 'Represent each value by its membership in previously learned ranges.'),
    'decode_quantile_response': ('Decode normalized ordered bin probabilities as the finite weighted midpoint expectation.', 'Convert probabilities over learned ranges back into an estimated value.'),
}



def packages():
    pending = ['numpy', 'pydantic']
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
    provider_repo = ROOT.parent/'sciona-atoms-ml'
    tests = [ROOT/'tests/test_quantile_target_encoding.py', provider_repo/'tests/test_quantile_response_encoding_provider.py']
    paths = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'provider/'+p.name: p for p in tests}
    for relative in ('sciona/quantile_target_encoding.py', 'scripts/review_quantile_response_semantics.py',
                     'scripts/import_quantile_response_drafts.py', 'scripts/plan_quantile_response_intake.py',
                     'docs/licenses/mercari-solution-LICENSE.txt'):
        paths[relative] = ROOT/relative
    for module in (atoms, registry, abstract, dimensions):
        paths[module.__name__] = Path(inspect.getfile(module))
    before = {key: sha(path) for key, path in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='quantile-response-review-') as directory:
        xml = Path(directory)/'tests.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *map(str, tests), '--junitxml='+str(xml)],
                                cwd=ROOT, env=environment, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Provider qualification tests failed: '+result.stdout[-3000:]+result.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {key: sum(int(s.get(key, '0')) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=21, failures=0, errors=0, skipped=0):
        raise ValueError('All 21 focused qualification tests required')
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for item in intake['new_providers']:
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                actual = db.execute(f'''SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',
                    (item['atom_id'],)).fetchall()
                if actual != [dict(fqdn=item['runtime_fqdn'], version_id=item['version_id'], content_hash=item['content_hash'], trust_tier=3)]:
                    raise ValueError('Exact current provider version required')
                ports = db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (item['version_id'],)).fetchall()
                if ports != sorted(item['interfaces'], key=lambda p: (p['direction'], p['ordinal'])):
                    raise ValueError('Exact stored provider interfaces required')
            operation, plain = OPERATIONS[item['runtime_fqdn'].rsplit('.', 1)[1]]
            records.append(dict(artifact_id=item['atom_id'], fqdn=item['runtime_fqdn'], version_id=item['version_id'],
                content_hash=item['content_hash'], interfaces=sorted(item['interfaces'], key=lambda p:(p['direction'],p['ordinal'])),
                operation=operation, plain_description=plain,
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Domain-independent numeric target encoding with explicit response coordinate, bin count and precision policy. No competition field names or fixed response transform are required.',
                registered_provider_dependencies=[]))
    if before != {key: sha(path) for key,path in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Software or qualification inputs changed during review')
    return dict(format='quantile-response-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=records,
        tests=counts, software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(PLAN), connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1, test_workers=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/quantile_response_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=result['tests'],
                         packages=len(result['installed_packages']), approved=False)))
