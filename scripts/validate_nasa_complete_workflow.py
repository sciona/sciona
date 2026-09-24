"""Compare the complete intake candidate with the pinned corrected source oracle."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost
from threadpoolctl import threadpool_limits

from sciona.nasa_complete_workflow_graph import build_complete_workflow_graph,select_complete_workflow_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
from scripts.audit_nasa_original_intake_scope import audit,ROOT
import scripts.validate_nasa_domain_graph as oracle


def validate(source):
    scope=audit(source)
    paths=[Path(__file__),ROOT/'sciona/nasa_complete_workflow_graph.py',ROOT/'scripts/audit_nasa_original_intake_scope.py']
    pins={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    original_execute=runner.CDGExecutionSession.execute;slots={}
    async def record(session,payload,**kwargs):
        previous=runner.save_intermediate_value;captured={}
        def capture(path,node,name,value):
            if (node,name)==('domain_output','out_prediction_table'):captured['output']=value.copy()
            if (node,name) in [('offsets','out_offsets'),('retained','out_selection'),('split','out_train')]:
                captured[(node,name)]=value.copy()
            previous(path,node,name,value)
        with patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=await original_execute(session,payload,**kwargs)
        slots[payload['airport']]=dict(payload=payload,captured=captured)
        return result
    with patch.object(runner.CDGExecutionSession,'execute',record):
        comparison=oracle.validate(source)
    if not comparison['passed'] or len(slots)!=10:raise ValueError('All source configurations required')
    airports=tuple(slots)
    def combine(field):
        return {key:pd.concat([slot['payload'][field][key] for slot in slots.values()],ignore_index=True)
            for key in ['queries','estimates','stands','arrivals']}
    training=combine('training_records');query=combine('prediction_records')
    expected=pd.concat([slot['captured']['output'] for slot in slots.values()],ignore_index=True).set_index(['gufi','timestamp','airport'])
    controls={name:slots[airports[0]]['payload'][name] for name in ['train_fraction','seed','threshold','prediction_feature_name']}
    cases=[]
    for count in [10,2]:
        records=dict(query,queries=query['queries'].loc[query['queries'].airport.isin(tuple(reversed(airports))[:count])]
                     .sample(frac=1,random_state=910+count).reset_index(drop=True))
        payload=dict(training_records=training,prediction_records=records,airports=airports,
            maximum_errors={airport:slot['payload']['maximum_error'] for airport,slot in slots.items()},**controls)
        graph=select_complete_workflow_graph(payload)
        digest,nodes,edges=encode_execution_graph(graph)
        graph=decode_execution_graph(nodes,edges,digest)
        if digest!=encode_execution_graph(build_complete_workflow_graph(count))[0]:raise ValueError('Selected workflow differs')
        captured={};closed=False;fits=[]
        original_regressor_fit=xgboost.XGBRegressor.fit
        original_classifier_fit=xgboost.XGBClassifier.fit
        def guard(original,label):
            def fit(model,*args,**kwargs):
                if closed:raise AssertionError('No fitting is allowed after the complete training-state handoff')
                fits.append(label)
                return original(model,*args,**kwargs)
            return fit
        def capture(path,node,name,value):
            nonlocal closed
            if (node,name)==('training_exhausted','out_result'):
                if len(value)!=10:raise ValueError('Complete model-state population required')
                closed=True
            if (node,name)==('inference_population_output','out_prediction_table'):captured['output']=value.copy()
            for index,airport in enumerate(airports):
                for identity,port in [('offsets','out_offsets'),('retained','out_selection'),('split','out_train')]:
                    if (node,name)==(f'training_p{index:02d}_{identity}',port):
                        np.testing.assert_array_equal(value,slots[airport]['captured'][(identity,port)])
                        captured[(index,identity)]=True
        with tempfile.TemporaryDirectory(prefix='nasa-complete-workflow-') as temporary:
            with threadpool_limits(limits=1),patch.object(runner,'RUNS_DIR',Path(temporary)), \
                    patch.object(runner,'save_intermediate_value',side_effect=capture), \
                    patch.object(xgboost.XGBRegressor,'fit',guard(original_regressor_fit,'regressor')), \
                    patch.object(xgboost.XGBClassifier,'fit',guard(original_classifier_fit,'classifier')):
                result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-complete-workflow',str(count)).execute(payload,cdg=graph))
        if result['status']!='completed' or not closed or fits.count('regressor')!=20 or fits.count('classifier')!=10:
            raise ValueError('Complete source fitting lifecycle required')
        if len([key for key in captured if isinstance(key,tuple)])!=30:raise ValueError('All populations, masks and offsets must match')
        output=captured['output']
        ordered=expected.loc[pd.MultiIndex.from_frame(records['queries'][['gufi','timestamp','airport']])].reset_index()
        pd.testing.assert_frame_equal(output,ordered)
        cases.append(dict(active_query_populations=count,training_populations=10,nodes=len(graph.nodes),edges=len(graph.edges),
            graph_sha256=digest,predictions=len(output),regressor_fits=20,classifier_fits=10,
            source_predictions_exact=True,source_masks_and_offsets_exact=True,no_fitting_after_state_handoff=True,
            original_query_order_preserved=True))
        print(json.dumps(cases[-1]),flush=True)
    if pins!={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}:
        raise ValueError('Complete workflow validation code drift')
    return dict(passed=True,approved=False,synthetic_only=True,catalog_mutations=0,cases=cases,
        source_oracle_predictions=comparison['output_rows'],predictions_compared=sum(case['predictions'] for case in cases),
        implementation_sha256=pins,shared_source_sha256=comparison['execution_source_sha256'],
        original_scope_audit=scope,
        limitations=['A corrected executable intake candidate, not approval of the old placeholder template.',
            'Full and sparse complete-workflow cases execute here; all ten inference regions were separately qualified in the approved population family.',
            'Runtime records and state replace notebook, filesystem-discovery and pickle orchestration; no historical backend or empirical-quality claim.',
            'Original-intake versioning, immutable provenance checks, catalog qualification and promotion remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path,required=True)
    args=parser.parse_args()
    report=validate(args.source_directory)
    (ROOT/'docs/reviews/nasa_complete_workflow_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(report['cases']),predictions=report['predictions_compared'],approved=False,catalog_mutations=0)))
