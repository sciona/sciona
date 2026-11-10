"""Qualify portable sparse neural providers for automated Tier 3 publication."""
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
from sciona.atoms.ml import sparse_neural_prediction as atoms
from sciona.ghost import registry, abstract, dimensions
from scripts.import_sparse_neural_drafts import reviewed_plan, ROOT, PLAN, sha
from scripts.import_flavours_provider_drafts import connect

LIMITS = [
    'Automated Tier 3 only; no human Tier 1 review or Tier 2 usage certification.',
    'Synthetic source parity does not establish accuracy or generalization on new domains. Caller owns training-only fitting, feature adapters, frozen vocabulary and normalized feature/response coordinates.',
    'Exact feature identities and coordinate labels are required but do not authenticate semantics. No implicit log transform, inverse transform or clipping is performed.',
    'MSE and quantile classification are distinct objectives. First-layer regularization, stored-entry binary semantics, partial batches and explicit learning-rate/batch schedules follow the declared implementation.',
    'Numeric JSON state restores known prediction graphs only. Optimizer state and training resumption are excluded. Learned weights and fitted decoder state are private runtime material.',
    'Abstract witnesses validate metadata, not numerical values or sparse storage. Known physical dimensions require upstream normalization; unknown dimensions are allowed.',
    'Each call uses a fresh synchronous single-core CPU worker; callers must include it in the shared resource budget. TensorFlow initializes before other numeric frameworks inside that worker.',
    'Installed files match their own RECORD hashes and active dependency constraints. Independent package provenance, clean deployment and other platforms are not certified.',
    'The connected source feature loaders remain diagnostic. This review qualifies two neural atoms, not the original competition CDG or completeness of the winning ensemble.',
]



def packages():
    pending = ['tensorflow', 'scikit-learn', 'numpy', 'scipy', 'pydantic']
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
    tests = [ROOT/'tests/test_sparse_neural_regression.py', ROOT/'tests/test_sparse_neural_state.py',
             provider_repo/'tests/test_sparse_neural_prediction_provider.py']
    paths = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'provider/'+p.name: p for p in tests}
    for relative in ('sciona/sparse_neural_regression.py', 'sciona/sparse_neural_state.py',
                     'sciona/sparse_neural_process.py', 'sciona/sparse_neural_worker.py',
                     'sciona/quantile_target_encoding.py', 'scripts/review_sparse_neural_semantics.py',
                     'scripts/import_sparse_neural_drafts.py', 'scripts/plan_sparse_neural_intake.py',
                     'docs/licenses/mercari-solution-LICENSE.txt'):
        paths[relative] = ROOT/relative
    for module in (atoms, registry, abstract, dimensions):
        paths[module.__name__] = Path(inspect.getfile(module))
    before = {key: sha(path) for key, path in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
        TF_CPP_MIN_LOG_LEVEL='2', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='sparse-neural-review-') as directory:
        counts = dict(tests=0, failures=0, errors=0, skipped=0)
        # Core tests need TF-first initialization. Provider tests separately exercise
        # the ordinary sklearn-first parent and isolated worker import order.
        for index, selected in enumerate((tests[:2], tests[2:])):
            xml = Path(directory)/f'tests-{index}.xml'
            args = ['-q', *map(str, selected), '--junitxml='+str(xml)]
            command = ([str(ROOT/'.venv/bin/python'), '-c',
                'import sciona.sparse_neural_regression; import pytest; '
                'raise SystemExit(pytest.main('+repr(args)+'))'] if index == 0 else
                [str(ROOT/'.venv/bin/python'), '-m', 'pytest', *args])
            result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
            if result.returncode:
                raise ValueError('Synthetic provider tests failed: '+result.stdout[-3000:]+result.stderr[-1000:])
            suites = list(ET.parse(xml).getroot().iter('testsuite'))
            for key in counts:
                counts[key] += sum(int(s.get(key, '0')) for s in suites)
    if counts != dict(tests=29, failures=0, errors=0, skipped=0):
        raise ValueError('All 29 focused qualification tests required')
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
            fitting = item['runtime_fqdn'].endswith('.fit_sparse_neural')
            records.append(dict(artifact_id=item['atom_id'], fqdn=item['runtime_fqdn'], version_id=item['version_id'],
                content_hash=item['content_hash'], interfaces=sorted(item['interfaces'], key=lambda p:(p['direction'],p['ordinal'])),
                operation='Fit a sparse neural MSE or quantile predictor under explicit training controls, returning numeric prediction state.' if fitting else 'Restore a known sparse neural graph from validated numeric state and predict without training.',
                plain_description='Learn a numeric response from sparse training features.' if fitting else 'Predict a numeric response using a previously fitted sparse neural model.',
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Domain-independent sparse supervised regression with explicit ordered feature identities, normalized feature and response coordinates, and training controls. Cross-disciplinary use requires compatible caller-owned feature adapters; no competition schema or fixed response transform is required.',
                registered_provider_dependencies=[]))
    if before != {key: sha(path) for key,path in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Software or qualification inputs changed during review')
    return dict(format='sparse-neural-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=records,
        tests=counts, software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(PLAN), connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1, test_workers=1, synchronous_model_workers=1, test_suites_run_serially=True), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/sparse_neural_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=result['tests'],
                         packages=len(result['installed_packages']), approved=False)))
