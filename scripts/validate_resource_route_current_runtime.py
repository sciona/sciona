"""Offline source-bound resource-route lifecycle qualification; no catalog approval."""
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
from sciona.resource_route_graph import build_resource_route_graph
import sciona.atoms.ml.resource_route_execution as adapters
from scripts.review_video_numeric_runtime import review as runtime_review

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT.parent/'sciona-atoms-ml'
TESTS = ['tests/test_resource_route_planning.py', 'tests/test_resource_route_reuse.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    names = {node.matched_primitive for node in build_resource_route_graph().nodes}
    found = {}
    for spec in _parse_registered_atoms(repo=ProviderRepo(ML.name, ML), artifact_root=ML/'src/sciona/atoms'):
        name = spec.import_module+'.'+spec.source_symbol
        if name not in names:
            continue
        if name in found:
            raise ValueError('Ambiguous provider source')
        found[name] = dict(runtime_fqdn=name, catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
            content_hash=spec.content_hash, source_file=os.path.relpath(spec.file_path, ROOT),
            provider_sha256=sha(spec.file_path), role='generic_route_stage')
    if set(found) != names:
        raise ValueError('Missing registered provider source')
    return [found[name] for name in sorted(found)]


def validate():
    providers = source_inventory()
    paths = sorted(set(TESTS + ['sciona/resource_route_planning.py', 'sciona/resource_route_graph.py',
        'scripts/resource_route_synthetic.py', 'scripts/validate_resource_route_current_runtime.py',
        'scripts/review_video_numeric_runtime.py', 'tests/test_text_pair_provider_graphs.py',
        'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py']
        + [p['source_file'] for p in providers]))
    before = {p:sha(ROOT/p) for p in paths}
    runtime = runtime_review()
    graph_hashes = {'route':encode_execution_graph(build_resource_route_graph())[0]}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-route-runtime-') as temp:
        xml = Path(temp)/'tests.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
            '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Runtime qualification failed: '+result.stdout[-4000:]+result.stderr[-1000:])
        counts = {k:sum(int(s.get(k, '0')) for s in ET.parse(xml).getroot().iter('testsuite'))
            for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=21, failures=0, errors=0, skipped=0):
        raise ValueError('Complete passing execution required')
    if before != {p:sha(ROOT/p) for p in paths} or providers != source_inventory() or runtime != runtime_review():
        raise ValueError('Qualified software or runtime changed')
    return dict(format='resource-route-current-runtime.v1', passed=True, approved=False,
        catalog_mutations=0, catalog_state_checked=False, synthetic_only=True, tests=counts,
        test_paths=TESTS, software_sha256=before, providers=providers, runtime=runtime, graph_sha256=graph_hashes,
        native_route_cases=4, independently_compared_multiresource_graphs=60,
        independently_compared_single_resource_graphs=30, reuse_transformations=4,
        resource_policy=dict(serial=True, native_threads=1),
        limits=['Offline runtime qualification; exact catalog contracts and original publication remain pending.',
            'Finite directed nonnegative integer additive costs and bounded resources only.',
            'Caller supplies costs, resource units and closure meaning; no domain-specific ingestion.',
            'Trusted in-process intermediate objects; no portable state serialization or arbitrary forged-object qualification.',
            'Provisioned runtime package closure and file hashes checked; no clean-install or other-platform claim.'])


if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/resource_route_current_runtime.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), catalog_state_checked=False)))
