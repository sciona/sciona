"""Rerun unchanged synthetic VSB tests and graph without replacing old evidence."""
import contextlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET
from threadpoolctl import threadpool_info
from scripts.audit_decomposed_vsb_readiness import ROOT,sha,audit


def validate():
    readiness=audit()
    if readiness['implementation_drift']:raise ValueError('Reference implementation changed')
    tests=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_vsb_*.py'))
    paths={p:ROOT/p for p in tests}
    for p in (ROOT/'sciona').glob('vsb_*.py'):paths[str(p.relative_to(ROOT))]=p
    for p in ('scripts/validate_vsb_graph_execution.py','scripts/vsb_synthetic.py',
              'scripts/validate_vsb_current_baseline.py','scripts/audit_decomposed_vsb_readiness.py'):
        paths[p]=ROOT/p
    old_reports={p:sha(ROOT/'docs/reviews'/p) for p in readiness['retained_evidence_sha256']}
    before={p:sha(v) for p,v in paths.items()}
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='vsb-baseline-tests-') as directory:
        xml=Path(directory)/'tests.xml'
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*tests,'--junitxml='+str(xml)],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        if run.returncode:raise ValueError('Synthetic tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=57,failures=0,errors=0,skipped=0):raise ValueError('Complete 57-test reference suite required: '+str(counts))
    print('57 synthetic component tests passed; running full 125-model serialized graph.',flush=True)
    # Retain the historical harness unchanged; intercept only its report write.
    report_path=ROOT/'docs/reviews/competition_vsb_graph_execution.json'
    captured=[];original_write=Path.write_text;original_path=sys.path.copy()
    def capture(path,data,*args,**kwargs):
        if path.resolve()==report_path.resolve():
            captured.append(json.loads(data));return len(data)
        return original_write(path,data,*args,**kwargs)
    try:
        sys.path.insert(0,str(ROOT/'scripts'))
        spec=importlib.util.spec_from_file_location('vsb_retained_graph_harness',ROOT/'scripts/validate_vsb_graph_execution.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(Path,'write_text',capture),contextlib.redirect_stdout(io.StringIO()):module.main()
    finally:sys.path[:]=original_path
    if len(captured)!=1:raise ValueError('Exactly one completed graph report required')
    graph=captured[0]
    old=json.loads(report_path.read_text())
    for key in ('serialized_graph_sha256','provider_sha256','code_sha256','checks'):
        if graph[key]!=old[key]:raise ValueError('Historical graph coverage changed: '+key)
    if before!={p:sha(v) for p,v in paths.items()} or old_reports!={p:sha(ROOT/'docs/reviews'/p) for p in old_reports}:
        raise ValueError('Reference software or historical evidence changed')
    if readiness!=audit():raise ValueError('Catalog or implementation changed during execution')
    pools=threadpool_info()
    if any(p['num_threads']!=1 for p in pools):raise ValueError('Native thread budget exceeded')
    return dict(format='vsb-current-baseline.v1',passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        tests=counts,graph=graph,code_sha256=before,historical_evidence_sha256=old_reports,
        runtime_versions={n:metadata.version(n) for n in ('numpy','scipy','pandas','lightgbm','scikit-learn','threadpoolctl')},
        resource_policy=dict(serial_tests=True,serial_models=True,native_threads=1),
        limits=['Current execution of unchanged full reference only; decomposed providers and corrected original approval remain pending.',
                'All 125 models and full configured training ceilings exercised on synthetic populations; no historical accuracy or unbiased validation claim.',
                'Current runtime qualification does not overwrite or reinterpret historical dependency versions.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/vsb_current_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=result['tests'],graph_checks=result['graph']['checks'],approved=False)))
