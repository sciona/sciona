"""Actual 25-node runner compared with the corrected pinned source sequence."""
import argparse,asyncio,contextlib,hashlib,io,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from scripts.validate_nasa_corrected_workflow import source_programs,synthetic_tables
from scripts import validate_nasa_corrected_workflow as generation
from sciona.nasa_workflow_inputs import prepare_training_inputs,prepare_prediction_inputs,attach_predictions
from sciona.nasa_feature_adapters import airline_features
from sciona.tabular_contracts import stable_feature_order,ordered_feature_matrix
from sciona.residual_classifier_graph import build_residual_classifier_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner

ROOT=Path(__file__).resolve().parents[1]

def source_level_tables(rng,airport,training,vocabulary=None):
    records={}
    original_etd=generation.estimated_departure_features
    original_history=generation.arrival_history_features
    def etd(queries,estimates):
        records['estimates']=estimates.copy()
        return original_etd(queries,estimates)
    def history(times,stands,arrivals,destination):
        records['stands']=stands.copy();records['arrivals']=arrivals.copy()
        return original_history(times,stands,arrivals,destination)
    with patch.object(generation,'estimated_departure_features',side_effect=etd),patch.object(generation,'arrival_history_features',side_effect=history):
        tables,vocabulary=synthetic_tables(rng,airport,training=training,vocabulary=vocabulary)
    records['queries']=tables['labels'].copy()
    return tables,vocabulary,records

def validate(source):
    controls,programs,source_hashes=source_programs(source)
    digest,nodes,edges=encode_execution_graph(build_residual_classifier_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    cases=[]
    for index,airport in enumerate(controls['list_airports']):
        rng=np.random.default_rng(9021+index)
        training,vocabulary,training_records=source_level_tables(rng,airport,True)
        evaluation,_,prediction_records=source_level_tables(rng,airport,False,vocabulary)
        def loader(tables):
            def read(path,**kwargs):
                if 'train_labels_' in path or path.endswith('submission_data.csv'):return tables['labels'].copy()
                for suffix,key in [('_etd.csv','etd'),('_airlinecode.csv','codes'),('_taxitime_to_gate.csv','taxi')]:
                    if path.endswith(suffix):return tables[key].copy()
                raise ValueError('Unexpected source IO')
            return read
        ns=dict(np=np,pd=pd,xgb=xgb,GroupShuffleSplit=GroupShuffleSplit,mean_absolute_error=mean_absolute_error,mean_squared_error=mean_squared_error,airport=airport,raw_label_load_dir='',timepointgufi_root='',gufi_root='',timepoint_root='',stable_feature_order=stable_feature_order,**controls)
        with threadpool_limits(limits=1),patch.object(pd,'read_csv',loader(training)),contextlib.redirect_stdout(io.StringIO()):exec(programs[0],ns)
        offsets={k:ns[k] for k in ['median_underestimation','median_overestimation']}
        models=dict(regressor=ns['regressor_lower'],classifier=ns['estimate_classifier'],classifier_params=offsets)
        def encode(frame,unused):return frame.merge(airline_features(frame,vocabulary).drop(columns='airport'),on='gufi',how='left',validate='many_to_one')
        inference=dict(np=np,pd=pd,airport=airport,model={airport:models},debug=False,raw_label_load_dir='',timepointgufi_root_submission='',timepoint_root_submission='',grab_airlinecodes=encode)
        with threadpool_limits(limits=1),patch.object(pd,'read_csv',loader(evaluation)),contextlib.redirect_stdout(io.StringIO()):exec(programs[1],inference)
        names=stable_feature_order(ns['feature_cols']);frame=ns['df_data']
        _,groups=np.unique(frame.gufi.to_numpy(),return_inverse=True)
        cutoff=controls['mae_thresh_bad'] if airport in controls['bad_airports'] else controls['mae_thresh_good']
        payload=dict(features=ordered_feature_matrix(frame,names),targets=frame.minutes_until_pushback.to_numpy(dtype=np.float64),groups=groups.astype(np.int64),feature_names=names,prediction_features=ordered_feature_matrix(inference['df_predict'],names),train_fraction=.4,seed=42,maximum_error=float(cutoff),threshold=.5,prediction_feature_name='pred_minutes_until_pushback')
        prepared,learned_vocabulary,training_identities=prepare_training_inputs(training_records,airport)
        query_matrix,query_identities=prepare_prediction_inputs(prediction_records,airport,prepared['feature_names'],learned_vocabulary)
        for key in ['features','targets','groups']:np.testing.assert_array_equal(prepared[key],payload[key])
        assert prepared['feature_names']==names and learned_vocabulary==vocabulary
        np.testing.assert_array_equal(query_matrix,payload['prediction_features'])
        payload.update(prepared,prediction_features=query_matrix)
        captured={}
        def capture(directory,node,name,value):
            if (node,name) in [('final','out_predictions'),('offsets','out_offsets'),('retained','out_selection'),('split','out_train')]:captured[(node,name)]=value
        with tempfile.TemporaryDirectory(prefix='residual-classifier-graph-') as temporary:
            with threadpool_limits(limits=1),patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
                result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-residual-graph',str(index)).execute(payload,cdg=graph))
        if result['status']!='completed':raise ValueError('Graph did not complete')
        np.testing.assert_array_equal(captured[('final','out_predictions')],inference['df_predict'].minutes_until_pushback.to_numpy())
        np.testing.assert_array_equal(captured[('offsets','out_offsets')],list(offsets.values()))
        np.testing.assert_array_equal(np.flatnonzero(captured[('retained','out_selection')]),ns['df_internal_test_lower'].index.to_numpy())
        np.testing.assert_array_equal(np.flatnonzero(captured[('split','out_train')]),ns['train_index'])
        attached=attach_predictions(query_identities,captured[('final','out_predictions')])
        pd.testing.assert_frame_equal(attached,inference['df_predict'][list(attached)].reset_index(drop=True))
        case=dict(slot=index,cutoff=cutoff,completed=True,predictions=64,selected_populations_exact=True,conditional_offsets_exact=True,final_int32_predictions_exact=True,source_level_adapter_integration=True,output_identity_alignment_exact=True)
        cases.append(case);print(json.dumps(case),flush=True)
    return dict(passed=True,approved=False,synthetic_only=True,catalog_mutations=0,graph_sha256=digest,nodes=len(graph.nodes),edges=len(graph.edges),scenarios=cases,source_execution_ast_sha256=source_hashes,builder_sha256=hashlib.sha256((ROOT/'sciona/residual_classifier_graph.py').read_bytes()).hexdigest(),validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations=['Complete numerical graph on supplied feature matrices; domain feature adapters and multi-airport dispatch are not nodes in this graph.','Actual runner and codec execution; intermediate persistence intercepted so models remained runtime values.','Full ghost pass, precise published IO contracts, dependency/license review and catalog publication gates remain pending.'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);args=parser.parse_args()
    report=validate(args.source_directory)
    (ROOT/'docs/reviews/residual_classifier_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='scenarios'}))
