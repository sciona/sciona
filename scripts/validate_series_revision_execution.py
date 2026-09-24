"""Execute the selected series revision against independent circuit equations."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile

import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_series_revision import check
from scripts.stage_series_original_revision import ROOT,sha,require
from scripts.audit_series_original_scope import audit
from sciona.visualizer import runner


async def validate(source,approved=False):
    require(audit(source)==json.loads((ROOT/'docs/reviews/physics_series_original_scope.json').read_text()),'Fresh source scope differs')
    connection=dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL']
    with psycopg.connect(connection,row_factory=dict_row,options='-c default_transaction_read_only=on') as db:graph,imported=check(db,approved)
    rng=np.random.default_rng(20260910)
    a=10**rng.uniform(-3,4,256);b=10**rng.uniform(-3,4,256);i=rng.uniform(.01,2,256)*rng.choice([-1,1],256)
    matrix=np.array([[1.,0.,0.],[0.,1.,0.],[-1.,-1.,1.]])
    expected=np.linalg.solve(matrix,np.stack([i*a,i*b,np.zeros(256)]))[2]/i
    cases=[('kirchhoff_reference',a,b,i,expected),('permuted',b,a,i,expected),('reversed_current',a,b,-2*i,expected),
        ('broadcast',np.array([[1.],[2.]]),np.array([3.,4.,5.]),np.array(-.5),np.array([[4.,5.,6.],[5.,6.,7.]])),
        ('zero_resistance',np.array(0.),np.array(0.),np.array(1.),np.array(0.))]
    invalid=[('zero_current',1.,2.,0.,'current must be nonzero'),('negative_a',-1.,2.,1.,'resistance must be nonnegative'),
        ('negative_b',1.,-2.,1.,'resistance must be nonnegative'),('nan',np.nan,2.,1.,'finite real numeric inputs required'),
        ('infinite_current',1.,2.,np.inf,'finite real numeric inputs required'),('boolean',True,2.,1.,'finite real numeric inputs required'),
        ('complex',1+1j,2.,1.,'finite real numeric inputs required'),('text','1',2.,1.,'finite real numeric inputs required'),
        ('overflow',np.finfo(float).max,np.finfo(float).max,1.,'overflow encountered in add'),
        ('incompatible_shapes',np.ones(2),np.ones(3),1.,'shape mismatch')]
    results=[];rejected=[]
    with tempfile.TemporaryDirectory(prefix='sciona-series-revision-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for name,first,second,current,reference in cases:
                payload=dict(resistance_a=np.asarray(first),resistance_b=np.asarray(second),current=np.asarray(current))
                before={k:v.copy() for k,v in payload.items()}
                result=await runner.CDGExecutionSession(None,'synthetic-series',name).execute(payload,cdg=graph)
                require(result['status']=='completed','Stored graph execution failed')
                outputs=runner.load_cached_outputs(Path(directory)/name,'series_equivalent_resistance')
                require(outputs and 'out_equivalent_resistance' in outputs,'Stored output missing')
                raw=outputs['out_equivalent_resistance']
                actual=np.asarray(raw)
                require(actual.shape==reference.shape and actual.dtype==np.dtype('float64'),'Output contract differs')
                np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-12)
                for key,value in payload.items():np.testing.assert_array_equal(value,before[key])
                results.append(dict(case=name,passed=True,shape=list(actual.shape),values=int(actual.size),inputs_unmodified=True,
                    stored_output_kind='array' if isinstance(raw,np.ndarray) else 'scalar',
                    maximum_absolute_error=float(np.max(np.abs(actual-reference)))))
            for name,first,second,current,message in invalid:
                try:await runner.CDGExecutionSession(None,'synthetic-series',name).execute(dict(resistance_a=np.asarray(first),resistance_b=np.asarray(second),current=np.asarray(current)),cdg=graph)
                except RuntimeError as error:
                    require(message in str(error),'Unexpected rejection: '+name)
                    require(not runner.load_cached_outputs(Path(directory)/name,'series_equivalent_resistance'),'Invalid case produced output')
                    rejected.append(name)
                else:raise ValueError('Invalid input accepted: '+name)
        finally:runner.RUNS_DIR=prior
    with psycopg.connect(connection,row_factory=dict_row,options='-c default_transaction_read_only=on') as db:
        after,after_imported=check(db,approved)
        require(after==graph and after_imported==imported,'Catalog changed during execution')
    return dict(passed=True,approved=approved,catalog_mutations=0,version_id=imported['version_id'],graph_sha256=imported['graph_sha256'],
        independent_circuit_cases=256,valid_cases=results,invalid_cases_rejected=rejected,reused_atoms=1,new_atoms=0,original_history_preserved=True,
        scope='Stored revision through CDGExecutionSession; independent Kirchhoff linear-system reference, permutation/current invariance, broadcasting, zero resistance and invalid-domain rejection.',
        limitations=['Two ohmic components in a series branch are caller-supplied physical assumptions. Numerical arrays cannot establish topology.',
            'Synthetic tests do not qualify arbitrary physical systems, HTTP transport, clean installation or redistribution.'],
        validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_series_revision.py'),
        source_sha256={name:sha(ROOT/name) for name in ['sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']},scope_audit_sha256=imported['scope_audit_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--approved',action='store_true')
    args=parser.parse_args();report=asyncio.run(validate(args.source_directory,args.approved))
    name='series_revision_served_execution.json' if args.approved else 'series_revision_catalog_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=args.approved,independent_circuit_cases=256,invalid_rejections=len(report['invalid_cases_rejected']))))
