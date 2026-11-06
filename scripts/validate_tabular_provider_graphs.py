"""Bind direct numeric native executions to immutable intake and semantic evidence."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.tabular_provider_graphs import PROVIDERS, build_provider_graph
from scripts.stage_complete_tabular_providers import ROOT, reviewed, stage, sha, require
from scripts.review_video_numeric_runtime import review as runtime_review


def validate():
    intake = reviewed()
    require(stage()['rows_created'] == 0, 'Exact staged providers required')
    semantic_path = 'docs/reviews/complete_tabular_provider_semantics.json'
    raw = (ROOT/semantic_path).read_bytes()
    require(raw == subprocess.check_output(['git', 'show', '0125379e:'+semantic_path], cwd=ROOT), 'Immutable semantic evidence changed')
    semantic = json.loads(raw)
    for path, digest in semantic['software_sha256'].items(): require(sha(ROOT/path) == digest, 'Semantic sources changed')
    paths = ['sciona/tabular_provider_graphs.py', 'tests/test_tabular_provider_graphs.py',
        'tests/test_text_pair_provider_graphs.py', 'scripts/validate_tabular_provider_graphs.py', semantic_path]
    before = {p:sha(ROOT/p) for p in paths}
    runtime = runtime_review()
    graphs = {name:encode_execution_graph(build_provider_graph(name))[0] for name in PROVIDERS}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-tabular-native-') as temp:
        xml = Path(temp)/'tests.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', paths[1],
            '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=env, capture_output=True, text=True)
        require(result.returncode == 0, 'Native checks failed: '+result.stdout[-4000:])
        counts = {k:sum(int(s.get(k,'0')) for s in ET.parse(xml).getroot().iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    require(counts == dict(tests=15, failures=0, errors=0, skipped=0), 'All native cases required')
    require(before == {p:sha(ROOT/p) for p in paths} and reviewed() == intake and runtime_review() == runtime,
        'Qualification sources or runtime changed')
    require(stage()['rows_created'] == 0, 'Staged catalog changed')
    return dict(format='tabular-provider-native-qualification.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, tests=counts, uncached_native_executions=16,
        graph_sha256=graphs, software_sha256=before, runtime=runtime,
        intake_plan_sha256=semantic['intake_plan_sha256'], semantic_review_sha256=sha(ROOT/semantic_path),
        resource_policy=dict(serial=True, native_threads=1),
        limits=['Six reusable numeric providers executed independently of recipe adapters.',
            'All nine group-reduction branches checked; a native-fitted frequency map reused in a separate query execution.',
            'Every test serializes and decodes its graph and requires uncached completed traces.',
            'Native parity supplements independent numerical semantic checks; no publication is performed.'])


if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/tabular_provider_native_qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], executions=result['uncached_native_executions'], approved=False)))
