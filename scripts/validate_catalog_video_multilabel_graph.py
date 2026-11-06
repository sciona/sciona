"""Reexecute the complete sharded lifecycle using exact catalog contracts."""
import json
import os
import subprocess
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.freeze_video_graph_contracts import ROOT, PATH, freeze, sha
from sciona.catalog_video_multilabel_graph import build_catalog_video_multilabel_graph
from sciona.services.execution_graph_codec import encode_execution_graph


def validate():
    frozen=json.loads(PATH.read_text())
    if freeze()!=frozen:raise ValueError('Frozen catalog contracts changed')
    prior_path='docs/reviews/decomposed_video_multilabel_graph_execution_v4.json'
    raw=(ROOT/prior_path).read_bytes()
    if raw!=subprocess.check_output(['git','show','8673d9be:'+prior_path],cwd=ROOT):raise ValueError('Prior evidence changed')
    prior=json.loads(raw)
    for path,digest in prior['software_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Qualified software changed')
    paths=list(prior['software_sha256'])+[
        'scripts/freeze_video_graph_contracts.py','scripts/validate_catalog_video_multilabel_graph.py',
        'sciona/catalog_video_multilabel_graph.py','tests/test_catalog_video_multilabel_graph.py',
        'docs/reviews/video_graph_contracts.json']
    before={p:sha(ROOT/p) for p in paths}
    graph=build_catalog_video_multilabel_graph()
    digest=encode_execution_graph(graph)[0]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-catalog-video-') as temporary:
        xml=Path(temporary)/'tests.xml'
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q','tests/test_catalog_video_multilabel_graph.py',
            '--junitxml='+str(xml)],cwd=ROOT,env=env,capture_output=True,text=True)
        if run.returncode:raise ValueError('Catalog graph validation failed: '+run.stdout[-4000:]+run.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=3,failures=0,errors=0,skipped=0):raise ValueError('Three full lifecycle cases required')
    if before!={p:sha(ROOT/p) for p in paths} or freeze()!=frozen:raise ValueError('Qualified source or catalog changed')
    return dict(format='catalog-video-multilabel-execution.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,tests=counts,graph_sha256=digest,software_sha256=before,
        prior_execution_evidence_sha256=sha(ROOT/prior_path),graph_nodes=len(graph.nodes),explicit_callable_edges=len(graph.edges),
        exact_catalog_ports=True,exact_registered_bindings=True,all_nodes_uncached=True,full_configured_lifecycle_parity=True,
        approved_numeric_providers=9,draft_recipe_providers=1,resource_policy=dict(serial_tests=True,native_threads=1),
        limits=['Three complete synthetic executions, including 80 epochs; no historical competition accuracy claim.',
                'Recipe and original CDG approval remain pending. Earlier 100-test numerical evidence is source-pinned, not reported as rerun here.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/catalog_video_multilabel_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=result['tests'],nodes=result['graph_nodes'],approved=False)))
