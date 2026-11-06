"""Fingerprint concrete primitive candidates and the full synthetic pair lifecycle."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.atoms.ml.text_nlp.atoms import clean_text, levenshtein
from sciona.atoms.ml.text_nlp.bm25 import score_bm25
from sciona.atoms.ml.sklearn.isotonic.atoms import (
    isotonic_regression, isotonic_regression_fit, isotonic_regression_transform,
)
from scripts.review_video_numeric_runtime import review as runtime_review

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT.parent / 'sciona-atoms-ml'
TESTS = ['tests/test_bm25_provider.py', 'tests/test_text_pair_fields.py',
         'tests/test_text_pair_complete_contract.py', 'tests/test_text_pair_complete_recipe.py',
         'tests/test_text_pair_bm25.py', 'tests/test_text_pair_normalization_policy.py',
         '../sciona-atoms-ml/tests/test_ml_sklearn_isotonic_behavior.py',
         '../sciona-atoms-ml/tests/test_ml_text_nlp_behavior.py::test_clean_text_removes_url_and_corrects_spelling',
         '../sciona-atoms-ml/tests/test_ml_text_nlp_behavior.py::test_string_similarity_atoms']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    sources = ['sciona/bm25_scoring.py', 'sciona/text_pair_bm25.py', 'sciona/text_pair_fields.py',
        'sciona/text_pair_complete_contract.py', 'sciona/text_pair_complete_recipe.py',
        'sciona/text_pair_normalization_policy.py', 'sciona/text_pair_features.py',
        'sciona/text_pair_training.py', 'sciona/text_pair_contract.py', 'sciona/sparse_randomized_projection.py',
        'sciona/materialized_prediction.py', 'tests/test_text_pair_training.py',
        'scripts/validate_complete_text_pair_providers.py', 'scripts/review_video_numeric_runtime.py']
    provider_files = ['text_nlp/atoms.py', 'text_nlp/witnesses.py', 'text_nlp/bm25.py',
        'sklearn/isotonic/atoms.py', 'sklearn/isotonic/state_models.py', 'sklearn/isotonic/witnesses.py',
        'sklearn/decomposition/sparse_projection.py', 'model_selection/materialized_prediction.py',
        'calibration/empirical_f1.py']
    paths = sorted(set(sources + [t.split('::')[0] for t in TESTS]
                       + ['../sciona-atoms-ml/src/sciona/atoms/ml/' + p for p in provider_files]))
    before = {p: sha(ROOT / p) for p in paths}
    runtime = runtime_review()
    wanted = {f.__module__ + '.' + f.__name__: f for f in (
        clean_text, levenshtein, score_bm25, isotonic_regression, isotonic_regression_fit, isotonic_regression_transform)}
    records = []
    for spec in _parse_registered_atoms(repo=ProviderRepo(ML.name, ML), artifact_root=ML / 'src/sciona/atoms'):
        name = spec.import_module + '.' + spec.source_symbol
        if name in wanted:
            records.append(dict(runtime_fqdn=name, catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
                content_hash=spec.content_hash, provider_sha256=sha(spec.file_path),
                signature=str(inspect.signature(wanted[name]))))
    if len(records) != len(wanted) or {r['runtime_fqdn'] for r in records} != set(wanted):
        raise ValueError('Exact six provider source identities required')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-text-pair-') as temp:
        xml = Path(temp) / 'tests.xml'
        result = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
            '-W', 'error::RuntimeWarning', '--junitxml=' + str(xml)], cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Synthetic provider validation failed: ' + result.stdout[-6000:] + result.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=115, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 115-test suite required: ' + str(counts))
    if before != {p: sha(ROOT / p) for p in paths} or runtime != runtime_review():
        raise ValueError('Qualified source or runtime changed during validation')
    return dict(format='complete-text-pair-provider-qualification.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, tests=counts,
        providers=sorted(records, key=lambda r: r['runtime_fqdn']), software_sha256=before,
        runtime=runtime, resource_policy=dict(serial=True, native_threads=1),
        limits=['Concrete current-platform qualification; catalog intake, semantic promotion and native graph execution remain pending.',
                'Existing text witnesses are not proved value-preserving abstractions; no abstract-value equivalence claim.',
                'Isotonic validation covers the declared increasing clip calibration branch and listed provider behavioral cases, not every possible control/state.',
                'Source fingerprints cover exercised algorithm modules and tests, not every transitively imported package module.',
                'No real datasets, historical competition accuracy or empirical probability-calibration claim.',
                'The existing event-loop fixture deprecation is unrelated to numerical execution.'])


if __name__ == '__main__':
    result = validate()
    (ROOT / 'docs/reviews/complete_text_pair_provider_qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
