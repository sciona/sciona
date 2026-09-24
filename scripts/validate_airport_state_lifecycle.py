"""Reload complete synthetic learned state in fresh inference-only processes."""
import argparse
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from sciona.residual_model_state import pack_state,encode_state,decode_state
from sciona.visualizer import runner

ROOT=Path(__file__).resolve().parents[1]


def child(directory):
    import xgboost
    from sciona.airport_model_state import predict_records
    def forbidden(*args,**kwargs):raise AssertionError('Fresh inference must not refit models')
    xgboost.XGBRegressor.fit=forbidden
    xgboost.XGBClassifier.fit=forbidden
    state=decode_state((directory/'state.json').read_bytes())
    records={key:pd.read_json(io.StringIO(value),orient='table') for key,value in json.loads((directory/'queries.json').read_text()).items()}
    output=predict_records(records,state)
    np.save(directory/'predictions.npy',output.minutes_until_pushback.to_numpy(),allow_pickle=False)
    if not output[['gufi','timestamp','airport']].reset_index(drop=True).equals(records['queries'][['gufi','timestamp','airport']].reset_index(drop=True)):
        raise ValueError('Fresh inference identity order differs')


def validate(source):
    import scripts.validate_nasa_domain_graph as domain
    original_execute=runner.CDGExecutionSession.execute
    cases=[]
    async def execute(session,payload,**kwargs):
        captured={}
        expected={('lower_fit','out_state'):'regressor',('classifier_fit','out_state'):'classifier',
            ('offsets','out_offsets'):'offsets',('domain_vocabulary','out_vocabulary'):'vocabulary',
            ('domain_train_arrays','out_feature_names'):'feature_names',('final','out_predictions'):'predictions'}
        original_capture=runner.save_intermediate_value
        def capture(directory,node,name,value):
            if (node,name) in expected:captured[expected[node,name]]=value
            original_capture(directory,node,name,value)
        with patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=await original_execute(session,payload,**kwargs)
        if result['status']!='completed' or len(captured)!=6:
            raise ValueError('Complete learned-state capture required')
        state=pack_state(captured['feature_names'],payload['prediction_feature_name'],captured['regressor'],
            captured['classifier'],captured['offsets'],payload['threshold'],
            dict(format='sciona.airport.feature-state.v1',airport=payload['airport'],
                 vocabulary=list(captured['vocabulary']),target_unit='minutes'))
        with tempfile.TemporaryDirectory(prefix='airport-state-lifecycle-') as temporary:
            directory=Path(temporary)
            (directory/'state.json').write_bytes(encode_state(state))
            records={key:value.to_json(orient='table',date_format='iso',date_unit='ns') for key,value in payload['prediction_records'].items()}
            (directory/'queries.json').write_text(json.dumps(records))
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
            completed=subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child-directory',str(directory)],
                cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if completed.returncode:
                raise ValueError('Fresh inference process failed: '+completed.stderr[-2000:])
            actual=np.load(directory/'predictions.npy',allow_pickle=False)
            np.testing.assert_array_equal(actual,captured['predictions'])
        cases.append(dict(slot=len(cases),predictions=len(actual),fresh_process=True,no_refit=True,
                          native_state_reloaded=True,raw_query_features_recomputed=True,exact_predictions=True))
        print(json.dumps(cases[-1]),flush=True)
        return result
    with patch.object(runner.CDGExecutionSession,'execute',execute):
        source_result=domain.validate(source)
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,scenarios=cases,
        source_graph_sha256=source_result['graph_sha256'],source_predictions=sum(case['predictions'] for case in cases),
        implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['sciona/residual_model_state.py','sciona/airport_model_state.py','scripts/validate_airport_state_lifecycle.py']},
        limitations=['Fresh-process synthetic state persistence and inference; no real records or model fixtures retained.',
                    'Exact installed backend version only; digests establish corruption detection, not authenticity.',
                    'Training/inference graph separation and multi-airport graph publication remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path)
    parser.add_argument('--child-directory',type=Path)
    args=parser.parse_args()
    if args.child_directory:child(args.child_directory)
    elif args.source_directory:
        report=validate(args.source_directory)
        (ROOT/'docs/reviews/competition_nasa_state_lifecycle.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=True,scenarios=len(report['scenarios']),predictions=report['source_predictions'])))
    else:parser.error('A source directory or child directory is required')
