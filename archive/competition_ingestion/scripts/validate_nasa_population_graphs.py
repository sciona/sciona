"""Synthetic source parity for explicit population training and sparse inference."""
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
from threadpoolctl import threadpool_limits

import sciona.atoms.ml.pipeline.keyed_routing as routing
import sciona.atoms.ml.domain_adapters.airport_population as population
import sciona.atoms.ml.domain_adapters.airport_state as state_provider
from sciona.nasa_population_graphs import build_population_graph,select_population_graph
from sciona.residual_model_state import encode_state,decode_state
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner

ROOT=Path(__file__).resolve().parents[1]


def decoded(kind,count):
    digest,nodes,edges=encode_execution_graph(build_population_graph(kind,count))
    return digest,decode_execution_graph(nodes,edges,digest)


def child(directory):
    import xgboost
    def forbidden(*args,**kwargs):raise AssertionError('Inference must not fit models')
    xgboost.XGBRegressor.fit=forbidden
    xgboost.XGBClassifier.fit=forbidden
    request=json.loads((directory/'request.json').read_text())
    states={key:decode_state(value.encode()) for key,value in json.loads((directory/'states.json').read_text()).items()}
    records={key:pd.read_json(io.StringIO(value),orient='table')
             for key,value in json.loads((directory/'queries.json').read_text()).items()}
    payload=dict(model_states=states,prediction_records=records)
    digest,nodes,edges=encode_execution_graph(select_population_graph('inference',payload))
    graph=decode_execution_graph(nodes,edges,digest)
    if graph.metadata['active_population_count']!=request['count']:
        raise ValueError('Selected active population count differs')
    captured={};branches=[]
    def capture(path,node,name,value):
        if (node,name)==('population_output','out_prediction_table'):captured['output']=value
        if node.startswith('p') and node.endswith('_domain_output') and name=='out_prediction_table':branches.append(node)
    with threadpool_limits(limits=1),patch.object(runner,'RUNS_DIR',directory/'runs'), \
            patch.object(runner,'save_intermediate_value',side_effect=capture):
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-population-inference','fresh').execute(
            payload,cdg=graph))
    if result['status']!='completed' or len(branches)!=request['count']:
        raise ValueError('Every active population must complete exactly one visible inference branch')
    output=captured['output']
    pd.testing.assert_frame_equal(output[['gufi','timestamp','airport']],records['queries'][['gufi','timestamp','airport']])
    np.save(directory/'predictions.npy',output.minutes_until_pushback.to_numpy(),allow_pickle=False)
    (directory/'result.json').write_text(json.dumps(dict(passed=True,no_refit=True,branches=len(branches),
        graph_sha256=digest,nodes=len(graph.nodes),rows=len(output),identity_alignment_exact=True)))


def validate(source):
    import scripts.validate_nasa_domain_graph as oracle
    paths={str(path.relative_to(ROOT)):path for path in [Path(__file__),ROOT/'sciona/nasa_population_graphs.py',
        ROOT/'sciona/nasa_population_dispatch.py',ROOT/'sciona/nasa_lifecycle_graphs.py',
        ROOT/'sciona/residual_model_state.py',ROOT/'sciona/airport_model_state.py']}
    paths.update(routing_provider=Path(routing.__file__),population_provider=Path(population.__file__),state_provider=Path(state_provider.__file__))
    pins={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    original_execute=runner.CDGExecutionSession.execute
    slots={}
    async def record(session,payload,**kwargs):
        previous=runner.save_intermediate_value
        captured={}
        def capture(path,node,name,value):
            if (node,name)==('domain_output','out_prediction_table'):captured['output']=value.copy()
            if (node,name)==('offsets','out_offsets'):captured['offsets']=value.copy()
            previous(path,node,name,value)
        with patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=await original_execute(session,payload,**kwargs)
        slots[payload['airport']]=dict(payload=payload,**captured)
        return result
    with patch.object(runner.CDGExecutionSession,'execute',record):
        comparison=oracle.validate(source)
    if not comparison['passed'] or len(slots)!=10:raise ValueError('Complete pinned source oracle required')
    airports=tuple(slots)
    def combine(field):
        return {key:pd.concat([slots[airport]['payload'][field][key] for airport in airports],ignore_index=True)
                for key in ['queries','estimates','stands','arrivals']}
    training_records=combine('training_records');prediction_records=combine('prediction_records')
    training_digest,training_graph=decoded('training',len(slots))
    control_names=['train_fraction','seed','threshold','prediction_feature_name']
    controls={name:slots[airports[0]]['payload'][name] for name in control_names}
    if any(any(slot['payload'][name]!=controls[name] for name in control_names) for slot in slots.values()):
        raise ValueError('Shared source controls differ')
    payload=dict(training_records=training_records,airports=airports,
        maximum_errors={airport:slot['payload']['maximum_error'] for airport,slot in slots.items()},**controls)
    if encode_execution_graph(select_population_graph('training',payload))[0]!=training_digest:
        raise ValueError('Selected training population count differs')
    captured={};branch_offsets={}
    def capture(path,node,name,value):
        if (node,name)==('exhausted','out_result'):captured['states']=value
        if node.endswith('_offsets') and name=='out_offsets':branch_offsets[node]=value
    with tempfile.TemporaryDirectory(prefix='nasa-population-training-') as temporary:
        with threadpool_limits(limits=1),patch.object(runner,'RUNS_DIR',Path(temporary)), \
                patch.object(runner,'save_intermediate_value',side_effect=capture):
            outcome=asyncio.run(runner.CDGExecutionSession(None,'synthetic-population-training','all').execute(payload,cdg=training_graph))
    if outcome['status']!='completed' or tuple(captured['states'])!=airports:
        raise ValueError('All source training populations must produce complete state')
    for index,airport in enumerate(airports):
        np.testing.assert_array_equal(branch_offsets[f'p{index:02d}_offsets'],slots[airport]['offsets'])
    expected=pd.concat([slot['output'] for slot in slots.values()],ignore_index=True)
    expected=expected.set_index(['gufi','timestamp','airport'])
    cases=[]
    for count in range(1,len(airports)+1):
        # Reverse slot selection and shuffle rows independently of model order.
        selected=tuple(reversed(airports))[:count]
        queries=prediction_records['queries']
        queries=queries.loc[queries.airport.isin(selected)].sample(frac=1,random_state=700+count).reset_index(drop=True)
        records=dict(prediction_records,queries=queries)
        with tempfile.TemporaryDirectory(prefix='nasa-population-inference-') as temporary:
            directory=Path(temporary)
            (directory/'states.json').write_text(json.dumps({key:encode_state(value).decode() for key,value in captured['states'].items()}))
            (directory/'queries.json').write_text(json.dumps({key:value.to_json(orient='table',date_format='iso',date_unit='ns') for key,value in records.items()}))
            (directory/'request.json').write_text(json.dumps(dict(count=count)))
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
            process=subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child-directory',str(directory)],
                cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if process.returncode:raise ValueError('Fresh population inference failed: '+process.stderr[-2000:])
            case=json.loads((directory/'result.json').read_text())
            predictions=np.load(directory/'predictions.npy',allow_pickle=False)
            matched=expected.loc[pd.MultiIndex.from_frame(queries[['gufi','timestamp','airport']])].minutes_until_pushback.to_numpy()
            np.testing.assert_array_equal(predictions,matched)
            if case['graph_sha256']!=decoded('inference',count)[0] or not case['no_refit']:
                raise ValueError('Fresh inference graph identity differs')
        cases.append(dict(case,active_populations=count,source_predictions_exact=True,unqueried_populations_skipped=10-count))
        print(json.dumps(cases[-1]),flush=True)
    if pins!={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}:
        raise ValueError('Population implementation drift during validation')
    return dict(passed=True,approved=False,synthetic_only=True,catalog_mutations=0,
        training_graph_sha256=training_digest,training_nodes=len(training_graph.nodes),training_populations=len(slots),
        training_without_queries=True,source_calibration_offsets_exact=True,source_oracle_predictions=comparison['output_rows'],
        inference_cases=cases,fresh_processes=len(cases),predictions_compared=sum(case['rows'] for case in cases),
        source_sha256=pins,shared_source_sha256=comparison['execution_source_sha256'],
        limitations=['Explicit finite graph variants require active count selection before execution; exhaustion rejects count mismatches.',
            'Nonempty query batches only; training requires every declared population. No implicit missing-model fallback.',
            'Synthetic behavior qualification only; static table propagation, catalog promotion and original-intake completion remain pending.',
            'All models, query tables and intermediate files are private temporary runtime values and are deleted.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path)
    parser.add_argument('--child-directory',type=Path)
    args=parser.parse_args()
    if args.child_directory:child(args.child_directory)
    elif args.source_directory:
        report=validate(args.source_directory)
        (ROOT/'docs/reviews/competition_nasa_population_graphs.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:value for key,value in report.items() if key not in ('inference_cases','source_sha256','shared_source_sha256')}))
    else:parser.error('Source or child directory required')
