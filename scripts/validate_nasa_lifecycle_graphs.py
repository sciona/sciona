"""Compare separate training/fresh-process inference graphs with the source oracle."""
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

import sciona.atoms.ml.domain_adapters.airport_state as state_provider
from sciona.nasa_lifecycle_graphs import build_nasa_training_graph,build_nasa_inference_graph
from sciona.residual_model_state import encode_state,decode_state
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner

ROOT=Path(__file__).resolve().parents[1]


def decoded(builder):
    digest,nodes,edges=encode_execution_graph(builder())
    return digest,decode_execution_graph(nodes,edges,digest)


def child(directory):
    import xgboost
    def forbidden(*args,**kwargs):raise AssertionError('Inference graph must not fit models')
    xgboost.XGBRegressor.fit=forbidden
    xgboost.XGBClassifier.fit=forbidden
    digest,graph=decoded(build_nasa_inference_graph)
    state=decode_state((directory/'state.json').read_bytes())
    records={key:pd.read_json(io.StringIO(value),orient='table') for key,value in json.loads((directory/'queries.json').read_text()).items()}
    captured={}
    def capture(path,node,name,value):
        if (node,name)==('domain_output','out_prediction_table'):captured['output']=value
    with patch.object(runner,'RUNS_DIR',directory/'runs'),patch.object(runner,'save_intermediate_value',side_effect=capture):
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-inference','fresh').execute(
            dict(model_state=state,prediction_records=records),cdg=graph))
    if result['status']!='completed' or 'output' not in captured:raise ValueError('Inference graph failed')
    output=captured['output']
    pd.testing.assert_frame_equal(output[['gufi','timestamp','airport']].reset_index(drop=True),
                                  records['queries'][['gufi','timestamp','airport']].reset_index(drop=True))
    np.save(directory/'predictions.npy',output.minutes_until_pushback.to_numpy(),allow_pickle=False)
    (directory/'result.json').write_text(json.dumps(dict(graph_sha256=digest,completed=True,no_refit=True,rows=len(output))))


def validate(source):
    import scripts.validate_nasa_domain_graph as oracle
    training_digest,training_graph=decoded(build_nasa_training_graph)
    inference_digest,inference_graph=decoded(build_nasa_inference_graph)
    original_execute=runner.CDGExecutionSession.execute
    source_paths={name:ROOT/name for name in ['sciona/nasa_lifecycle_graphs.py','sciona/residual_model_state.py',
        'sciona/airport_model_state.py','scripts/validate_nasa_lifecycle_graphs.py']}
    source_paths['state_provider']=Path(state_provider.__file__)
    pins={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in source_paths.items()}
    cases=[]
    async def execute(session,payload,**kwargs):
        training_payload={name:payload[name] for name in ['training_records','airport','train_fraction','seed',
                          'maximum_error','threshold','prediction_feature_name']}
        original_capture=runner.save_intermediate_value
        captured={}
        def capture(directory,node,name,value):
            if (node,name)==('learned_state','out_model_state'):captured['state']=value
            original_capture(directory,node,name,value)
        with patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=await original_execute(session,training_payload,cdg=training_graph)
        if result['status']!='completed' or 'state' not in captured:raise ValueError('Training graph failed')
        with tempfile.TemporaryDirectory(prefix='nasa-lifecycle-graphs-') as temporary:
            directory=Path(temporary)
            (directory/'state.json').write_bytes(encode_state(captured['state']))
            records={key:value.to_json(orient='table',date_format='iso',date_unit='ns') for key,value in payload['prediction_records'].items()}
            (directory/'queries.json').write_text(json.dumps(records))
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
            process=subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child-directory',str(directory)],
                cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if process.returncode:raise ValueError('Fresh inference graph failed: '+process.stderr[-2000:])
            verification=json.loads((directory/'result.json').read_text())
            if verification['graph_sha256']!=inference_digest or not verification['no_refit']:
                raise ValueError('Fresh inference identity differs')
            predictions=np.load(directory/'predictions.npy',allow_pickle=False)
        # The source oracle consumes the independently executed graph outputs.
        # Child identity equality was verified before exporting the aligned vector.
        output=payload['prediction_records']['queries'][['gufi','timestamp','airport']].copy().reset_index(drop=True)
        output['minutes_until_pushback']=predictions
        original_capture(session.run_dir,'final','out_predictions',predictions)
        original_capture(session.run_dir,'domain_output','out_prediction_table',output)
        cases.append(dict(slot=len(cases),predictions=len(predictions),training_without_queries=True,
                          fresh_inference_without_refitting=True,identity_alignment_exact=True))
        print(json.dumps(cases[-1]),flush=True)
        return result
    with patch.object(runner.CDGExecutionSession,'execute',execute):
        comparison=oracle.validate(source)
    if pins!={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in source_paths.items()}:
        raise ValueError('Lifecycle implementation changed during execution')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        training_graph_sha256=training_digest,inference_graph_sha256=inference_digest,
        training_nodes=len(training_graph.nodes),inference_nodes=len(inference_graph.nodes),
        scenarios=cases,source_comparisons=comparison['source_scenarios'],source_predictions=sum(case['predictions'] for case in cases),
        lifecycle_source_sha256=pins,shared_execution_source_sha256=comparison['execution_source_sha256'],
        limitations=['Separate graphs execute in provisioned processes; all retained evidence is synthetic and aggregate.',
            'Training metadata uses its observed matrix as a witness shape probe; unseen query populations are checked when inference runs.',
            'Whole-graph static table propagation and multi-airport graph composition remain unsupported or pending.',
            'New lifecycle atoms and graphs are not yet catalog-approved.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path)
    parser.add_argument('--child-directory',type=Path)
    args=parser.parse_args()
    if args.child_directory:child(args.child_directory)
    elif args.source_directory:
        report=validate(args.source_directory)
        (ROOT/'docs/reviews/competition_nasa_lifecycle_graphs.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=True,training_nodes=report['training_nodes'],inference_nodes=report['inference_nodes'],predictions=report['source_predictions'])))
    else:parser.error('A source directory or child directory is required')
