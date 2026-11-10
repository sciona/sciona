"""Current-runtime contract checks; full Contrails lifecycle remains a separate gate."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from scripts.audit_contrails_original_readiness import ROOT, sha
from scripts import review_video_numeric_runtime as runtime_checker

TESTS = [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_contrails_*.py'))
         if p.name != 'test_contrails_publication_contract.py']
ROOT_PACKAGES = ['numpy', 'torch', 'torchvision', 'timm', 'segmentation-models-pytorch',
                 'scikit-learn', 'threadpoolctl']


def validate():
    paths = TESTS + [str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('contrails_*.py'))] + [
        'scripts/validate_contrails_current_contracts.py', 'scripts/review_video_numeric_runtime.py',
        '../sciona-atoms-dl/src/sciona/atoms/dl/contrails_execution.py']
    before = {p: sha(ROOT/p) for p in paths}
    previous_roots = runtime_checker.ROOT_PACKAGES
    runtime_checker.ROOT_PACKAGES = ROOT_PACKAGES
    try:
        runtime = runtime_checker.review()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                   MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
        with tempfile.TemporaryDirectory(prefix='synthetic-contrails-contracts-') as temporary:
            xml = Path(temporary)/'tests.xml'
            result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
                '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=env,
                capture_output=True, text=True)
            if result.returncode:
                raise ValueError('Contract qualification failed: '+result.stdout[-4000:]+result.stderr[-1000:])
            counts = {k: sum(int(s.get(k, '0')) for s in ET.parse(xml).getroot().iter('testsuite'))
                      for k in ('tests', 'failures', 'errors', 'skipped')}
        if counts != dict(tests=67, failures=0, errors=0, skipped=0):
            raise ValueError('Complete contract suite required')
        if runtime != runtime_checker.review() or before != {p:sha(ROOT/p) for p in paths}:
            raise ValueError('Qualified software changed')
    finally:
        runtime_checker.ROOT_PACKAGES = previous_roots
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        tests=counts, test_paths=TESTS, software_sha256=before, runtime=runtime,
        resource_policy=dict(serial_tests=True, native_threads=1),
        limits=['Component boundary, loss, fold, checkpoint, prediction and orchestration checks only; lifecycle tests use stand-in models.',
                'Historical publication-contract tests are separate and excluded; they validate old report identities rather than current full-model execution.',
                'Installed RECORD consistency does not authenticate publishers or qualify full source-model execution.',
                'Both v43 and v47 full source-model graphs still require current-runtime execution before original promotion.',
                'Historical immutable reports are preserved; no approval or predictive accuracy is claimed.'])



if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/contrails_current_contracts.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'])))
