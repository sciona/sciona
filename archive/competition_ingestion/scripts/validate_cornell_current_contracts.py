"""Current-runtime contract checks; full Cornell lifecycle remains a separate gate."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from scripts.restore_cornell_reviewed_software import ROOT, sha, verify
from sciona.cornell_augmentation import PINS, DEPENDENCIES, manifest
from scripts import review_video_numeric_runtime as runtime_checker

TESTS = ['tests/test_cornell_contract.py', 'tests/test_cornell_voting.py',
         'tests/test_cornell_training_controls.py']


def validate():
    source = manifest('competition_cornell_source_pins.json', PINS)
    dependencies = manifest('competition_cornell_historical_dependencies.json', DEPENDENCIES)
    verified = dict(source=verify(Path('/private/tmp/sciona_cornell_source'), source['pins']),
                    dependencies={d['wheel']: verify(Path('/private/tmp/sciona_cornell_dependencies') /
                        d['wheel'].split('-')[0], d['pins']) for d in dependencies})
    reviewed = dict(line.split('==') for line in (ROOT/'requirements/cornell-execution.txt').read_text().splitlines())
    installed = {name: metadata.version(name) for name in reviewed}
    paths = TESTS + [str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('cornell_*.py'))] + [
        'scripts/cornell_synthetic.py', 'scripts/validate_cornell_current_contracts.py',
        'scripts/restore_cornell_reviewed_software.py', 'scripts/review_video_numeric_runtime.py',
        'requirements/cornell-execution.txt']
    before = {p: sha((ROOT/p).read_bytes()) for p in paths}
    previous_roots = runtime_checker.ROOT_PACKAGES
    runtime_checker.ROOT_PACKAGES = sorted(set(previous_roots) | set(reviewed))
    try:
        runtime = runtime_checker.review()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                   MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
        with tempfile.TemporaryDirectory(prefix='synthetic-cornell-contracts-') as temporary:
            xml = Path(temporary)/'tests.xml'
            result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
                '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=env,
                capture_output=True, text=True)
            if result.returncode:
                raise ValueError('Contract qualification failed: '+result.stdout[-4000:]+result.stderr[-1000:])
            counts = {k: sum(int(s.get(k, '0')) for s in ET.parse(xml).getroot().iter('testsuite'))
                      for k in ('tests', 'failures', 'errors', 'skipped')}
        if counts != dict(tests=31, failures=0, errors=0, skipped=0):
            raise ValueError('Complete contract suite required')
        if runtime != runtime_checker.review() or before != {p:sha((ROOT/p).read_bytes()) for p in paths}:
            raise ValueError('Qualified software changed')
    finally:
        runtime_checker.ROOT_PACKAGES = previous_roots
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        tests=counts, test_paths=TESTS, verified_software_files=verified, software_sha256=before,
        runtime=runtime, prior_runtime_differences=[dict(package=n, reviewed=v, installed=installed[n])
            for n,v in reviewed.items() if installed[n] != v],
        resource_policy=dict(serial_tests=True, native_threads=1),
        limits=['Boundary, voting and checkpoint controls only; checkpoint tests use synthetic stand-in models.',
                'Installed RECORD consistency does not authenticate publishers or qualify full source-model execution.',
                'All thirteen DenseNet121 members, seventeen training phases, source augmentation and ten-copy inference still require current-runtime execution.',
                'Historical immutable reports are preserved; this report does not approve the original CDG.'])


if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/cornell_current_contracts.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], runtime_differences=result['prior_runtime_differences'])))
