"""Qualify serialized native sparse projection execution with frozen provider evidence."""
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
from sciona.atoms.ml.sklearn.decomposition import sparse_projection as provider
from scripts.review_video_numeric_runtime import review as runtime_review

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    paths=['sciona/sparse_randomized_projection.py','tests/test_sparse_randomized_projection.py',
        'tests/test_sparse_projection_provider.py','scripts/validate_sparse_projection_provider.py',
        'tests/test_sparse_projection_graph.py','sciona/sparse_projection_graph.py',
        'scripts/validate_sparse_projection_graph.py',
        '../sciona-atoms-ml/src/sciona/atoms/ml/sklearn/decomposition/sparse_projection.py']
    evidence='docs/reviews/sparse_projection_provider_qualification.json'
    raw=(ROOT/evidence).read_bytes()
    if raw!=subprocess.check_output(['git','show','fc26e4ae:'+evidence],cwd=ROOT):
        raise ValueError('Immutable registered-provider evidence changed')
    prior=json.loads(raw)
    for path,digest in prior['software_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Qualified source changed')
    paths.append(evidence)
    before={p:sha(ROOT/p) for p in paths}
    runtime=runtime_review()
    if runtime!=prior['runtime']:raise ValueError('Qualified runtime changed')
    directory=ROOT.parent/'sciona-atoms-ml'
    wanted={f.__module__+'.'+f.__name__:f for f in (provider.fit_sparse_projection,provider.transform_sparse_projection)}
    records=[]
    for spec in _parse_registered_atoms(repo=ProviderRepo(directory.name,directory),artifact_root=directory/'src/sciona/atoms'):
        name=spec.import_module+'.'+spec.source_symbol
        if name not in wanted:continue
        records.append(dict(runtime_fqdn=name,catalog_fqdn=spec.fqdn,version_id=str(spec.version_id),
            content_hash=spec.content_hash,provider_sha256=sha(spec.file_path),
            signature=str(inspect.signature(wanted[name])),contract=inspect.getdoc(wanted[name])))
    if len(records)!=2 or {r['runtime_fqdn'] for r in records}!=set(wanted):
        raise ValueError('Exactly two source-matched registered providers required')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-projection-') as temp:
        xml=Path(temp)/'tests.xml'
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q','tests/test_sparse_projection_graph.py',
            '--junitxml='+str(xml)],cwd=ROOT,env=env,capture_output=True,text=True)
        if run.returncode:raise ValueError('Provider validation failed: '+run.stdout[-5000:]+run.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=3,failures=0,errors=0,skipped=0):raise ValueError('Three complete native graph cases required')
    if before!={p:sha(ROOT/p) for p in paths} or runtime!=runtime_review():
        raise ValueError('Qualified implementation or runtime changed')
    return dict(format='sparse-projection-graph-qualification.v1',passed=True,approved=False,
        catalog_mutations=0,synthetic_only=True,tests=counts,providers=sorted(records,key=lambda r:r['runtime_fqdn']),
        software_sha256=before,runtime=runtime,prior_provider_evidence_sha256=sha(ROOT/evidence),
        native_executions=6,all_nodes_uncached=True,tuple_output_routing=True,frozen_query_reuse=True,resource_policy=dict(serial=True,native_threads=1),
        limits=['Current-platform synthetic numerical and retained lifecycle comparisons only.',
                'Six uncached native executions: three fit/project graphs and three query-only graphs with frozen factors. Prior 18-test provider evidence is source-pinned, not rerun here.',
                'No catalog approval, new text-pair feature policy or historical accuracy claim.',
                'Native graph execution runs with RuntimeWarning treated as error; direct registered outputs are compared exactly.',
                'Existing event-loop fixture deprecation remains unrelated to numerical qualification.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/sparse_projection_graph_qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=result['tests'],providers=len(result['providers']),approved=False)))
