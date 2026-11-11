"""Offline source-bound image_tabular lifecycle qualification; no catalog approval."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.image_tabular_complete_graph import build_complete_image_tabular_graph
from sciona.atoms.ml import image_tabular_complete_execution as adapters
from sciona.atoms.ml.calibration.empirical_f1 import fit_empirical_f1_thresholds, apply_column_thresholds
STAGES = ('prepare', 'fit', 'calibrate', 'predict')
PROVIDERS = dict(fit=fit_empirical_f1_thresholds, apply=apply_column_thresholds)
from scripts.review_video_numeric_runtime import review as runtime_review

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT.parent/'sciona-atoms-ml'
TESTS = sorted([str(p.relative_to(ROOT)) for pattern in ('test_image_tabular_*.py', 'test_empirical_f1_provider.py')
    for p in (ROOT/'tests').glob(pattern)])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    numeric = {fn.__module__+'.'+fn.__name__ for fn in PROVIDERS.values()}
    names = numeric | {adapters.__name__+'.'+name for name in STAGES}
    found = {}
    for spec in _parse_registered_atoms(repo=ProviderRepo(ML.name, ML), artifact_root=ML/'src/sciona/atoms'):
        name = spec.import_module+'.'+spec.source_symbol
        if name not in names:
            continue
        if name in found:
            raise ValueError('Ambiguous provider source')
        found[name] = dict(runtime_fqdn=name, catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
            content_hash=spec.content_hash, source_file=os.path.relpath(spec.file_path, ROOT),
            provider_sha256=sha(spec.file_path), role='numeric' if name in numeric else 'recipe_adapter')
    if set(found) != names:
        raise ValueError('Missing registered provider source')
    return [found[name] for name in sorted(found)]


def validate():
    providers = source_inventory()
    paths = sorted(set(TESTS + [str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('image_tabular_*.py')]
        + ['scripts/image_tabular_synthetic.py', 'scripts/validate_image_tabular_complete_runtime.py',
            'scripts/review_video_numeric_runtime.py', 'tests/test_text_pair_provider_graphs.py',
            'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py']
        + [p['source_file'] for p in providers]))
    before = {p:sha(ROOT/p) for p in paths}
    runtime = runtime_review()
    graph_hashes = {'recipe':encode_execution_graph(build_complete_image_tabular_graph())[0]}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-image_tabular-runtime-') as temp:
        xml = Path(temp)/'tests.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
            '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Runtime qualification failed: '+result.stdout[-4000:]+result.stderr[-1000:])
        counts = {k:sum(int(s.get(k, '0')) for s in ET.parse(xml).getroot().iter('testsuite'))
            for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=63,failures=0,errors=0,skipped=0):
        raise ValueError('Complete passing execution required')
    if before != {p:sha(ROOT/p) for p in paths} or providers != source_inventory() or runtime != runtime_review():
        raise ValueError('Qualified software or runtime changed')
    return dict(format='image_tabular-complete-runtime.v1', passed=True, approved=False,
        catalog_mutations=0, catalog_state_checked=False, synthetic_only=True, tests=counts,
        test_paths=TESTS, software_sha256=before, providers=providers, runtime=runtime, graph_sha256=graph_hashes,
        native_recipe_cases=3, native_numeric_cases=0, resource_policy=dict(serial=True, native_threads=1),
        limits=['Offline execution evidence only; original source and current catalog state still require reconciliation.',
            'Two reused domain-independent F1 providers and four group-separated image/tabular recipe adapters; numeric providers are exercised directly and within the native recipe.',
            'Fixed visual descriptors and fold-local metadata transforms feed a retained binary neural ensemble; held-out groups select the decision threshold. No independent accuracy or throughput claim.',
            'Trusted in-process intermediate objects only; no forged-object or portable serialization qualification.'],
        next_actions=['Verify immutable original source and retained derivative against the catalog.',
            'Review provider semantics, reconcile both catalogs, and qualify exact served bindings.',
            'Stage and execute the corrected original before Tier 3 publication and served retrieval checks.'])


if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/image_tabular_complete_runtime.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), catalog_state_checked=False)))
