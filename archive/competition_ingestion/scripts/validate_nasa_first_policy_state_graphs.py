"""Verify graph-trained model/policy state through complete guarded inference."""
import argparse
import ast
import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from catboost import CatBoostRegressor
import pandas as pd
from sciona.architect.handoff import CDGExport
from sciona.nasa_first_state_graphs import build_nasa_first_training_state_graph,build_nasa_first_state_inference_graph
from sciona.nasa_first_prediction import nasa_first_prediction
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.synthetic_nasa_first_workflow_inputs import population_inputs
from scripts.validate_nasa_first_native_wrapper import source_function,TRAIN_SOURCE,TRAIN_PIN
from scripts.validate_nasa_first_model_handoff import ROOT,sha


def validate(source,checkpoint):
    native_path=ROOT/'docs/reviews/competition_nasa_first_native_training_graph.json'
    native=json.loads(native_path.read_text())
    if not native['passed'] or json.loads((checkpoint/'qualification.json').read_text())!=native:
        raise ValueError('Qualified graph-trained models required')
    models={}
    for fit in native['fits']:
        index=fit['model_index'];path=checkpoint/f'model_{index}.cbm'
        if sha(path)!=fit['model_sha256']:raise ValueError('Model integrity differs')
        model=CatBoostRegressor(thread_count=1);model.load_model(str(path));models[index]=model
    from sciona.atoms.ml.domain_adapters.first_place_populations import POPULATIONS
    bank={label.upper():{0:models[2*i],1:models[2*i+1],2:models[2*i],'global_model':models[20]} for i,label in enumerate(POPULATIONS)}
    residual=source_function(source,TRAIN_SOURCE,TRAIN_PIN,'train_catboost_diff')
    cutoff=next(n for n in ast.walk(residual) if isinstance(n,ast.Compare) and isinstance(n.ops[0],ast.Gt))
    start=pd.Timestamp(cutoff.comparators[0].value)+pd.Timedelta(days=1)
    raw={}
    for i,label in enumerate(POPULATIONS):
        _,_,queries,options,policy=population_inputs(i,start)
        raw[label]=dict(queries=queries,raw_options=options,**policy)
    training=build_nasa_first_training_state_graph()
    tail_ids={'encode_bank','capture_policy','bind_state'}
    tail=CDGExport(nodes=[n for n in training.nodes if n.node_id in tail_ids],
        edges=[e for e in training.edges if e.source_id in tail_ids and e.target_id in tail_ids],
        metadata=dict(scope='State binding tail using separately qualified graph-trained models.'))
    inference=build_nasa_first_state_inference_graph()
    digests={}
    for name,graph in [('training',training),('tail',tail),('inference',inference)]:
        digest,nodes,edges=encode_execution_graph(graph)
        if decode_execution_graph(nodes,edges,digest)!=graph:raise ValueError('State graph codec differs')
        digests[name]=digest
    counter=0
    with tempfile.TemporaryDirectory(prefix='sciona-bound-state-') as directory:
        def execute(graph,inputs,expected_error=None):
            nonlocal counter
            counter+=1;captured={}
            def capture(path,node,port,value):
                if port.startswith('out_'):captured[node+'/'+port[4:]]=value
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                try:result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-bound-state',str(counter)).execute(inputs,cdg=graph))
                except RuntimeError as error:
                    if expected_error is None or not isinstance(error.__context__,ValueError) or str(error.__context__)!=expected_error:raise
                    return
            if expected_error is not None or result['status']!='completed':raise ValueError('State graph outcome differs')
            return captured
        state=execute(tail,dict(bank=bank,raw_populations=raw))['bind_state/state']
        state=json.loads(json.dumps(state,allow_nan=False))
        compared=0;routes=[]
        for label,entry in raw.items():
            query=entry['queries'].iloc[128:]
            policy={name:entry[name] for name in ['feature_columns','numeric_fills','categorical_fills']}
            expected,route=nasa_first_prediction(query,bank[label.upper()],label,raw_options=entry['raw_options'],**policy)
            options={key:value for key,value in entry['raw_options'].items() if key not in ['vocabularies','holiday_midnights']}
            actual=execute(inference,dict(state=state,queries=query,population=label,raw_options=options))
            if actual['predict/route']!=route or route!='primary':raise ValueError('Bound native route differs')
            pd.testing.assert_frame_equal(actual['format/predictions'],expected);compared+=len(query)
        label=POPULATIONS[0];entry=raw[label];query=entry['queries'].iloc[128:]
        policy={name:entry[name] for name in ['feature_columns','numeric_fills','categorical_fills']}
        for options,selected in [({'etd':entry['raw_options']['etd']},query),({},query),({},query.iloc[:0])]:
            expected,route=nasa_first_prediction(selected,bank[label.upper()],label,raw_options=options,**policy)
            actual=execute(inference,dict(state=state,queries=selected,population=label,raw_options=options))
            if actual['predict/route']!=route:raise ValueError('Bound fallback route differs')
            pd.testing.assert_frame_equal(actual['format/predictions'],expected);routes.append(route)
        bad=copy.deepcopy(state);bad['payload']['policy']['target_unit']='synthetic-wrong-unit'
        execute(inference,dict(state=bad,queries=query,population=label,raw_options={}),expected_error='Model/policy binding integrity differs')
        changed=copy.deepcopy(entry['raw_options']['vocabularies']);changed['RUNWAYS'].append('synthetic-new-category')
        execute(inference,dict(state=state,queries=query,population=label,raw_options={'vocabularies':changed}),
                expected_error='Inference vocabulary differs from training policy')
        execute(inference,dict(state=state,queries=query,population=label,raw_options={'holiday_midnights':pd.DatetimeIndex(['2032-01-01'])}),
                expected_error='Inference calendar differs from training policy')
    providers=['model_selection/policy_state.py','domain_adapters/first_place_policy.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,native_threads=1,
        graph_sha256=digests,training_nodes=len(training.nodes),inference_outer_nodes=len(inference.nodes),
        exact_native_query_comparisons=compared,fallback_routes=routes,policy_faults_rejected=3,
        training_state_tail_executed=True,full_training_rerun=False,json_state_handoff=True,
        native_training_qualification_sha256=sha(native_path),
        implementation_sha256={name:sha(ROOT/name) for name in ['sciona/policy_bound_state.py','sciona/nasa_first_policy_state.py',
            'sciona/nasa_first_state_graphs.py','scripts/validate_nasa_first_policy_state_graphs.py','tests/test_policy_bound_state.py']},
        provider_sha256={name:sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'/name) for name in providers},
        limitations=['Training state encoding tail executes on qualified graph-trained models; raw preparation and fitting retain their compositional evidence.',
            'Vocabularies, fills, ordered features and calendar policy remain private runtime state when non-public.',
            'Policy drift and state corruption fail before prediction fallbacks; new policies require an explicit new bound state.',
            'Catalog binding, review and publication remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True);args=parser.parse_args()
    report=validate(args.source_directory,args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_policy_state_graphs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,native_query_comparisons=report['exact_native_query_comparisons'],policy_faults_rejected=3)))
