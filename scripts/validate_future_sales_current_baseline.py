"""Execute the unchanged complete baseline on the current isolated runtime."""
import asyncio
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from packaging.requirements import Requirement
from threadpoolctl import threadpool_info
from sciona.future_sales_graph import build_future_sales_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts.future_sales_synthetic import payload
from scripts.review_future_sales_original_scope import review
from scripts.audit_decomposed_future_sales_readiness import ROOT, sha


def validate():
    scope=review()
    old=json.loads((ROOT/'docs/reviews/competition_future_sales_graph_execution.json').read_text())
    paths=old['code_sha256']
    if any(sha(ROOT/p)!=h for p,h in paths.items()):raise ValueError('Unchanged reference required')
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/future_sales_execution.py'
    if sha(provider)!=old['provider_sha256']:raise ValueError('Reference provider changed')
    historical=dict(line.split('==') for line in (ROOT/'requirements/future-sales-execution.txt').read_text().splitlines())
    versions={name:metadata.version(name) for name in historical}
    for name in versions:
        for text in metadata.requires(name) or []:
            requirement=Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
            if not requirement.specifier.contains(metadata.version(requirement.name),prereleases=True):
                raise ValueError('Active dependency constraint unsatisfied: '+name)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise ValueError('Explicit one-thread parent and worker pools required')
    digest,nodes,edges=encode_execution_graph(build_future_sales_graph())
    if digest!=old['serialized_graph_sha256']:raise ValueError('Reference graph changed')
    graph=decode_execution_graph(nodes,edges,digest)
    runner._ensure_atoms_imported()
    cases=[]
    for seed,trials in [(92,3),(17,4),(61,5)]:
        data=payload()
        data['training_controls'].update(seed=seed,trials=trials)
        if seed!=92:
            for i,row in enumerate(data['transactions']):row['quantity']=((i*7+seed)%31)-4
            data['feature_controls'].update(lags=[1,3],rolling_windows=[2,3])
        captured={}
        def capture(directory,node,name,value):
            if node=='execute' and name=='out_result':captured['result']=value
        with tempfile.TemporaryDirectory(prefix='future-sales-generated-baseline-') as directory:
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
                status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-future-sales','baseline').execute({'payload':data},cdg=graph))
        if status['status']!='completed' or len(status['trace'])!=2 or any(t['cached'] for t in status['trace']):
            raise ValueError('Complete uncached baseline graph required')
        direct=subprocess.run([sys.executable,'-m','sciona.future_sales_worker'],cwd=ROOT,env=environment,
            input=json.dumps(data,allow_nan=False),capture_output=True,text=True,timeout=120,check=False)
        if direct.returncode:raise ValueError('Independent worker invocation failed')
        expected=json.loads(direct.stdout)
        actual=json.loads(json.dumps(captured['result'],allow_nan=False))
        if actual!=expected:raise ValueError('Complete graph and direct worker result differ')
        if len(actual['trials'])!=trials or (actual['training_rows'],actual['validation_rows'],actual['refit_rows'],actual['forecast_rows'])!=(40,16,56,8):
            raise ValueError('Configured lifecycle incomplete')
        if not 1<=actual['selected_rounds']<=data['training_controls']['max_rounds']:
            raise ValueError('Selected training round count invalid')
        cases.append(dict(seed=seed,trials=trials,all_nodes_uncached=True,complete_result_equal=True,
            actual_runner_nodes=2,training_rows=40,validation_rows=16,refit_rows=56,forecast_rows=8))
    if any(p['num_threads']!=1 for p in threadpool_info()):raise ValueError('Native pool budget exceeded')
    if any(sha(ROOT/p)!=h for p,h in paths.items()) or sha(provider)!=old['provider_sha256'] or versions!={n:metadata.version(n) for n in versions}:
        raise ValueError('Reference or runtime changed during execution')
    return dict(format='future-sales-current-baseline.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,serialized_graph_sha256=digest,cases=cases,
        unchanged_reference_code_sha256=paths,provider_sha256=sha(provider),runtime=versions,historical_runtime=historical,
        active_dependency_constraints_satisfied=True,source_scope_review_sha256=sha(ROOT/'docs/reviews/future_sales_original_scope.json'),
        code_sha256={p:sha(ROOT/p) for p in ('scripts/validate_future_sales_current_baseline.py','sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py')},
        resource_policy=dict(serial_cases=True,serial_workers=True,native_threads=1,lightgbm_threads=1,optuna_jobs=1),
        limits=['Complete configured synthetic lifecycle and native graph wiring only; no original approval or competitive accuracy claim.',
                'Direct and graph paths invoke the same unchanged implementation in separate fresh processes; this is wiring parity, not an independent mathematical oracle.',
                'Dedicated worker isolation is retained. The earlier combined-process native crash is not claimed resolved.',
                'Installed versions and dependency constraints are checked; installed-file integrity and clean deployment remain separate qualification.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/future_sales_current_baseline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(report['cases']),approved=False)))
