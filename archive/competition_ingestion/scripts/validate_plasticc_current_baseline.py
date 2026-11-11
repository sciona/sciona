"""Re-execute retained synthetic PLAsTiCC runtime without overwriting evidence."""
import contextlib
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from threadpoolctl import threadpool_info
from scripts.audit_decomposed_plasticc_readiness import ROOT, sha, audit


def validate():
    readiness = audit()
    if readiness['implementation_drift']:
        raise ValueError('Retained reference implementation changed')
    paths = {str(p.relative_to(ROOT)): p for p in (ROOT/'sciona').glob('plasticc_*.py')}
    for p in ('scripts/validate_plasticc_current_baseline.py', 'scripts/audit_decomposed_plasticc_readiness.py',
              'scripts/validate_plasticc_runtime.py', 'scripts/validate_plasticc_graph.py',
              'tests/test_plasticc_publication_contract.py'):
        paths[p] = ROOT/p
    before = {p: sha(v) for p, v in paths.items()}
    reports = readiness['retained_evidence_sha256']
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',
        PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='plasticc-baseline-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_plasticc_publication_contract.py', '--junitxml='+str(xml)],
            cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Retained publication checks failed: '+run.stdout[-2000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=6, failures=0, errors=0, skipped=0):
        raise ValueError('Complete six-test reference contract suite required')
    print('Reference contract checks passed; executing both weighting modes and serialized graph.', flush=True)
    old_path = sys.path.copy()
    try:
        sys.path.insert(0, str(ROOT/'scripts'))
        runtime = importlib.import_module('validate_plasticc_runtime')
        graph = importlib.import_module('validate_plasticc_graph')
        config = runtime.synthetic_payload()['config']
        if config['training_parameters']['n_jobs'] != 1:
            raise ValueError('Single-worker training required')
        graph.runner._ensure_atoms_imported()
        # Retained validators return reports; their CLI report writers are not called.
        # All inputs are generated synthetic fixtures. Suppress verbose library output.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            current_runtime = runtime.validate()
            current_graph = graph.validate()
    finally:
        sys.path[:] = old_path
    for name, current in [('competition_plasticc_runtime.json', current_runtime),
                          ('competition_plasticc_graph.json', current_graph)]:
        if current != json.loads((ROOT/'docs/reviews'/name).read_text()):
            raise ValueError('Retained runtime coverage or software changed: '+name)
    if before != {p: sha(v) for p, v in paths.items()}:
        raise ValueError('Software changed during execution')
    if reports != {p: sha(ROOT/'docs/reviews'/p) for p in reports} or readiness != audit():
        raise ValueError('Historical evidence or catalog changed')
    if any(pool['num_threads'] != 1 for pool in threadpool_info()):
        raise ValueError('Native thread budget exceeded')
    return dict(format='plasticc-current-baseline.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, tests=counts, runtime=current_runtime, graph=current_graph,
        implementation_sha256=before, historical_evidence_sha256=reports,
        runtime_versions={n: metadata.version(n) for n in
            ('numpy', 'scipy', 'pandas', 'lightgbm', 'scikit-learn', 'astropy', 'george', 'threadpoolctl')},
        resource_policy=dict(serial_validators=True, training_workers=1, native_threads=1),
        limits=['Current retained three-stage execution only; decomposition and original promotion remain pending.',
                'Synthetic configuration uses one augmentation, three folds and a60-tree ceiling; it executes all configured work, not historical training scale.',
                'Both flat and redshift weighting paths execute. Numerical parity does not establish generalization or general novelty detection.',
                'Historical reports and source versions remain unchanged. No observation records or runtime identifiers are persisted.'])


if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/plasticc_current_baseline.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], runtime=result['runtime']['checks'],
                         graph=result['graph']['checks'], approved=False)))
