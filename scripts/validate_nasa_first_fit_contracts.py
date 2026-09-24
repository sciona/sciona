"""Compare graph fit operands with the pinned source using a recording backend."""
import argparse
import ast
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from sciona.nasa_first_fit_graph import build_nasa_first_fit_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.validate_nasa_first_native_wrapper import source_function, TRAIN_SOURCE, TRAIN_PIN
from scripts.validate_nasa_first_model_handoff import ROOT, sha


def validate(source):
    recorded = {}
    class RecordingRegressor:
        def __init__(self, **parameters):recorded['parameters']=parameters
        def fit(self,x,y,**options):
            recorded.update(x_train=x.copy(),y_train=y.copy(),x_valid=options['eval_set'][0].copy(),
                y_valid=options['eval_set'][1].copy(),categorical_indices=options['cat_features'],
                early_stopping_rounds=options['early_stopping_rounds'],use_best_model=options['use_best_model'])
            self.feature_importances_=np.ones(x.shape[1]);return self
        def predict(self,x):return np.ones(len(x))
    functions = [source_function(source,TRAIN_SOURCE,TRAIN_PIN,name) for name in ['train_catboost','train_catboost_diff']]
    namespace = {'pd':pd,'CatBoostRegressor':RecordingRegressor}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'pinned_training_contract','exec'),namespace)
    outcomes=[]
    rng=np.random.default_rng(103)
    with tempfile.TemporaryDirectory(prefix='sciona-fit-contracts-') as directory:
        for residual in [False,True]:
            original=build_nasa_first_fit_graph(residual=residual)
            digest,nodes,edges=encode_execution_graph(original)
            graph=decode_execution_graph(nodes,edges,digest)
            if graph!=original:raise ValueError('Fit graph codec differs')
            for case in range(3):
                table=pd.DataFrame(dict(gufi=np.arange(12),timestamp=pd.date_range('2020-11-01',periods=12,freq='12h'),
                    target=rng.uniform(1,100,12),etd_time_till_est_dep=rng.uniform(1,30,12),
                    synthetic_cat_machine=['A','B']*6,synthetic_load=rng.normal(size=12)))
                table.index=[4,4,6,8,8,9,2,1,0,7,5,5]
                table.loc[table.index==8,'target']=0
                if case==1:table['target']=table['target'].astype(np.int64)
                if case==2:table=table.iloc[::-1]
                cutoff='2020-11-04'
                with contextlib.redirect_stdout(io.StringIO()):
                    namespace['train_catboost_diff' if residual else 'train_catboost'](table.copy(), 'target', cutoff)
                captured={}
                def capture(path,node,port,value):
                    if node=='split' and port.startswith('out_'):captured[port[4:]]=value
                    if node=='inputs' and port in ['out_parameters','out_early_stopping_rounds']:captured[port[4:]]=value
                with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture), \
                        contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-fit-contract',str(len(outcomes))).execute(
                        dict(table=table,target_name='target',end_train=cutoff),target_node_id='split',cdg=graph))
                if result['status']!='completed':raise ValueError('Fit operand graph failed')
                for name in ['x_train','x_valid']:pd.testing.assert_frame_equal(captured[name],recorded[name])
                for name in ['y_train','y_valid']:pd.testing.assert_series_equal(captured[name],recorded[name],check_dtype=False,check_names=False)
                if captured['categorical_indices']!=recorded['categorical_indices']:raise ValueError('Categorical schema differs')
                parameters=dict(recorded['parameters'])
                if parameters.pop('thread_count')!=-1:raise ValueError('Source thread setting differs')
                if captured['parameters']!=parameters or captured['early_stopping_rounds']!=60 or not recorded['use_best_model']:
                    raise ValueError('Source fit policy differs')
                outcomes.append(dict(residual=residual,case=case,graph_sha256=digest,passed=True,
                    training_rows=len(captured['x_train']),validation_rows=len(captured['x_valid'])))
    provider_names=['model_selection/named_regression.py','domain_adapters/first_place_training.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,cases=outcomes,
        fit_backend='recording stub for source operands; native core separately exercised by unit test',
        graph_execution_target='split',source_sha256={TRAIN_SOURCE:TRAIN_PIN},generic_training_tests_passed=7,
        implementation_sha256={name:sha(ROOT/name) for name in ['sciona/named_regression_training.py','sciona/nasa_first_fit_graph.py',
            'scripts/validate_nasa_first_fit_contracts.py','tests/test_named_regression_training.py']},
        provider_sha256={name:sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'/name) for name in provider_names},
        limitations=['Targets explicitly normalize to float64; values, row order and frame schema match source operands.',
            'Native thread_count is fixed to one, backend file writes disabled and verbose training output suppressed.',
            'Full source-sized graph fitting, score acceptance, feature importance and 21-fit assembly remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_fit_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,source_operand_cases=len(report['cases']),native_graph_fitting=False)))
