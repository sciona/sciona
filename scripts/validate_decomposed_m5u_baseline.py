"""Refresh existing uncertainty component evidence before graph extraction."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    tests = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_m5u_*.py'))
    paths = tests+sorted(str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('m5u_*.py'))
    paths += ['sciona/m5u_search_parameters.json',str(Path(__file__).resolve().relative_to(ROOT))]
    before = {p:sha(ROOT/p) for p in paths}
    package_names = [line.split('==')[0] for line in (ROOT/'requirements/m5u-execution.txt').read_text().splitlines() if line]
    versions = {name:metadata.version(name) for name in package_names}
    environment = dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='m5u-component-tests-') as temporary:
        junit = Path(temporary)/'results.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*tests,'--junitxml='+str(junit)],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        if result.returncode:
            raise ValueError('Synthetic component tests failed: '+result.stdout[-3000:]+result.stderr[-1000:])
        suites = ET.parse(junit).getroot().findall('testsuite')
        counts = {key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','errors','failures','skipped')}
        if counts!=dict(tests=102,errors=0,failures=0,skipped=0):
            raise ValueError('Complete 102-test baseline required: '+str(counts))
    if before!={p:sha(ROOT/p) for p in paths} or versions!={name:metadata.version(name) for name in package_names}:
        raise ValueError('Implementation or runtime changed during tests')
    return dict(format='decomposed-m5u-baseline-tests.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,**counts,sha256=before,runtime=versions,
        resource_policy=dict(native_threads=1,lightgbm_threads=1,search_workers=1,serial_tests=True),
        limits=['Current component regression evidence only. Complete decomposed graph execution, provider semantics and original-CDG publication remain pending.',
                'Includes boundary and schedule tests with mocks and small native learner cases; does not replace full-budget training and inference qualification.'])


if __name__=='__main__':
    report = validate()
    (ROOT/'docs/reviews/decomposed_m5u_baseline_tests.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','tests','errors','failures','skipped')}))
