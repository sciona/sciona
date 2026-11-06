"""Freeze native graph evidence against the prior 115-test qualification."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from scripts.validate_complete_text_pair_providers import ROOT, sha
from scripts.review_video_numeric_runtime import review as runtime_review
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.text_pair_provider_graphs import PROVIDERS, build_provider_graph, build_isotonic_state_graph


def validate():
    evidence = 'docs/reviews/complete_text_pair_provider_qualification.json'
    raw = (ROOT / evidence).read_bytes()
    if raw != subprocess.check_output(['git', 'show', '4f1ef5e5:' + evidence], cwd=ROOT):
        raise ValueError('Immutable 115-test evidence changed')
    prior = json.loads(raw)
    for path, digest in prior['software_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Qualified provider/recipe source changed')
    paths = sorted(set(prior['software_sha256']) | {evidence,
        'sciona/text_pair_provider_graphs.py', 'tests/test_text_pair_provider_graphs.py',
        'scripts/validate_text_pair_provider_graphs.py', 'sciona/services/execution_graph_codec.py',
        'sciona/visualizer/runner.py'})
    before = {p: sha(ROOT / p) for p in paths}
    runtime = runtime_review()
    if runtime != prior['runtime']:
        raise ValueError('Qualified runtime changed')
    graphs = {name: encode_execution_graph(build_provider_graph(name))[0] for name in PROVIDERS}
    graphs['fit_then_transform'] = encode_execution_graph(build_isotonic_state_graph())[0]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-provider-graphs-') as temp:
        xml = Path(temp) / 'tests.xml'
        run = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_text_pair_provider_graphs.py', '-W', 'error::RuntimeWarning', '--junitxml=' + str(xml)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Native graph qualification failed: ' + run.stdout[-6000:] + run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=9, failures=0, errors=0, skipped=0):
        raise ValueError('All nine native cases required')
    if before != {p: sha(ROOT / p) for p in paths} or runtime != runtime_review():
        raise ValueError('Source/runtime changed during execution')
    return dict(format='text-pair-provider-native-qualification.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, tests=counts, native_executions=12,
        providers=prior['providers'], graph_sha256=graphs, software_sha256=before, runtime=runtime,
        prior_evidence_sha256=sha(ROOT / evidence), all_nodes_uncached=True, fitted_state_identity_routing=True,
        frozen_query_reuse=True, resource_policy=dict(serial=True, native_threads=1),
        limits=['Six individual providers and three fit/transform plus query-only lifecycle cases.',
                'Concrete current-platform native execution only; no catalog intake, semantic approval or full original-CDG execution claim.',
                'The prior 115-test evidence is immutable and source-checked, not rerun here.',
                'Isotonic graph cases cover ties, constant scores and decreasing labels under increasing bounded clip policy.',
                'Existing witnesses are not proved value-preserving abstractions; native parity does not establish that claim.'])


if __name__ == '__main__':
    result = validate()
    (ROOT / 'docs/reviews/text_pair_provider_native_qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], native_executions=result['native_executions'], approved=False)))
