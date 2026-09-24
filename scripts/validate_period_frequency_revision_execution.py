"""Execute the stored original revision with independent synthetic measurements."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile

import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_period_frequency_revision import check
from scripts.stage_period_frequency_original_revision import ROOT,sha,require
from scripts.audit_period_frequency_original_scope import audit
from sciona.physics_ingest.period_frequency_validation import periodic_signal_measurements
from sciona.visualizer import runner


async def validate(source,approved=False):
    scope=audit(source)
    require(scope==json.loads((ROOT/'docs/reviews/physics_period_frequency_original_scope.json').read_text()),'Current source scope differs')
    connection=dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL']
    with psycopg.connect(connection,row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph,imported=check(db,approved)
    periods,expected=periodic_signal_measurements()
    cases=[('independent_periodic_measurements',periods,expected,1e-6),
        ('mechanical_cycles',np.array([.5,2.,4.]),np.array([2.,.5,.25]),0),
        ('electrical_cycles',np.array([[.001,.002],[.004,.008]]),np.array([[1000.,500.],[250.,125.]]),0),
        ('recurring_event_scalar',np.array(8.),np.array(.125),0),
        ('empty_batch',np.empty((0,2)),np.empty((0,2)),0)]
    invalid=[('zero',np.array([0.]),'finite positive period required'),
        ('negative',np.array([-1.]),'finite positive period required'),
        ('nan',np.array([np.nan]),'finite positive period required'),
        ('infinity',np.array([np.inf]),'finite positive period required'),
        ('boolean',np.array([True]),'real numeric period required'),
        ('complex',np.array([1+1j]),'real numeric period required'),
        ('text',np.array(['1']),'real numeric period required'),
        ('reciprocal_overflow',np.array([np.nextafter(0.,1.)]),'overflow encountered in reciprocal'),
        ('reciprocal_underflow',np.array([np.finfo(float).max]),'underflow encountered in reciprocal')]
    results=[];rejected=[]
    with tempfile.TemporaryDirectory(prefix='sciona-period-revision-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for name,values,reference,tolerance in cases:
                before=values.copy()
                result=await runner.CDGExecutionSession(None,'synthetic-period-revision',name).execute({'period_seconds':values},cdg=graph)
                require(result['status']=='completed','Stored execution failed')
                actual=np.load(Path(directory)/name/'frequency'/'out_frequency_hz.npy')
                require(actual.shape==values.shape and actual.dtype==np.dtype('float64'),'Output shape/dtype differs')
                np.testing.assert_array_equal(values,before)
                np.testing.assert_allclose(actual,reference,rtol=tolerance,atol=0)
                results.append(dict(case=name,values=int(values.size),shape=list(values.shape),passed=True,input_unmodified=True,
                    relative_tolerance=tolerance,maximum_relative_error=float(np.max(np.abs(actual-reference)/reference)) if actual.size else 0.))
            for name,values,message in invalid:
                try:await runner.CDGExecutionSession(None,'synthetic-period-revision',name).execute({'period_seconds':values},cdg=graph)
                except RuntimeError as error:
                    require(message in str(error),'Unexpected runtime rejection: '+name)
                    require(not (Path(directory)/name/'frequency'/'out_frequency_hz.npy').exists(),'Rejected input produced output')
                    rejected.append(name)
                else:raise ValueError('Invalid input accepted: '+name)
        finally:runner.RUNS_DIR=prior
    with psycopg.connect(connection,row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        after,after_imported=check(db,approved)
        require(after==graph and after_imported==imported,'Catalog changed during execution')
    return dict(passed=True,approved=approved,catalog_mutations=0,version_id=imported['version_id'],graph_sha256=imported['graph_sha256'],
        independent_measurement_cases=256,additional_synthetic_values=8,valid_cases=results,invalid_cases_rejected=rejected,
        reused_atoms=1,new_atoms=0,original_history_preserved=True,
        scope='Stored selected original-identity revision executed through CDGExecutionSession. Independent zero-crossing period and FFT frequency measurements, synthetic cross-domain scalar/vector/matrix contracts, and invalid-domain rejection.',
        limitations=['Synthetic regime validation does not establish effectiveness on arbitrary real signals or period estimation.',
            'No HTTP transport or clean-install qualification. Temporary synthetic records and output files are deleted.'],
        validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_period_frequency_revision.py'),
        source_sha256={name:sha(ROOT/name) for name in ['sciona/physics_ingest/period_frequency_validation.py','sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']},
        scope_audit_sha256=imported['scope_audit_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--approved',action='store_true')
    args=parser.parse_args();report=asyncio.run(validate(args.source_directory,args.approved))
    name='period_frequency_revision_served_execution.json' if args.approved else 'period_frequency_revision_catalog_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=args.approved,measurement_cases=256,invalid_cases_rejected=len(report['invalid_cases_rejected']))))
