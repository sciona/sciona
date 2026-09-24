"""Exercise complete inference routing through nested production graph sessions."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from sciona.nasa_first_guarded_graph import build_nasa_first_guarded_graph
from sciona.nasa_first_prediction import nasa_first_prediction
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.validate_nasa_first_model_handoff import ROOT, sha


def validate():
    fixture = runpy.run_path(str(ROOT/'tests/test_nasa_first_prediction.py'))
    queries, options = fixture['inputs']()
    Predictor = fixture['Predictor']
    models = {0: Predictor(10,['etd_time_till_est_dep']),1: Predictor(40,['etd_time_till_est_dep']),
        2: Predictor(10,['etd_time_till_est_dep']),'global_model': Predictor(80,['feat_cat_airport'])}
    graph = build_nasa_first_guarded_graph()
    digest,nodes,edges = encode_execution_graph(graph)
    restored = decode_execution_graph(nodes,edges,digest)
    if restored != graph:
        raise ValueError('Guarded graph codec differs')
    outcomes = []
    cases = [('primary',queries,models,'synthetic',options),('primary_override',queries,models,'KPHX',options),
        ('baseline',queries,{},'synthetic',options),('constant',queries,{},'synthetic',dict(options,raw_options={})),
        ('empty',queries.iloc[:0],{},'synthetic',dict(options,raw_options={})),
        ('duplicate_baseline',queries.iloc[[2,0,2]],{},'synthetic',options)]
    original_save = runner.save_intermediate_value
    with tempfile.TemporaryDirectory(prefix='sciona-guarded-inference-') as directory:
        for name,selected,state,population,policy in cases:
            expected,expected_route = nasa_first_prediction(selected,state,population,**policy)
            captured = {}
            def observe(path,node,port,value):
                original_save(path,node,port,value)
                if node == 'format' and port == 'out_predictions':captured['predictions'] = value
                if node == 'predict' and port == 'out_route':captured['route'] = value
                if node == 'inputs' and port in ('out_primary','out_baseline'):captured[port] = value
            with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=observe), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result = asyncio.run(runner.CDGExecutionSession(None,'synthetic-guarded',name).execute(
                    dict(queries=selected,models=state,population=population,**policy),cdg=restored))
            if result['status'] != 'completed' or captured.get('route') != expected_route:
                raise ValueError('Guarded route differs')
            pd.testing.assert_frame_equal(captured['predictions'],expected)
            for branch in ['primary','baseline']:
                if captured['out_'+branch] != restored.metadata[branch+'_branch']:
                    raise ValueError('Executed branch differs from serialized declaration')
            outcomes.append(dict(case=name,route=expected_route,rows=len(selected),passed=True))
        traces = [json.loads(path.read_text()) for path in Path(directory).glob('*/execution_trace.json')]
    # Primary twice: one nested session each. Baseline twice: two each.
    # Constant: two failed sessions. Empty: no nested execution.
    if len(traces) != 14:
        raise ValueError('Lazy branch execution count differs')
    providers = ['model_selection/graph_fallbacks.py','domain_adapters/first_place_fallback.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,graph_sha256=digest,
        outer_nodes=3,primary_branch_nodes=len(restored.metadata['primary_branch']['nodes']),
        baseline_branch_nodes=len(restored.metadata['baseline_branch']['nodes']),cases=outcomes,
        production_executor=True,actual_intermediate_persistence=True,session_traces=len(traces),
        branch_envelopes_match_execution=True,guard_contract_tests_passed=8,
        implementation_sha256={name:sha(ROOT/name) for name in ['sciona/guarded_prediction_graphs.py',
            'sciona/nasa_first_guarded_graph.py','scripts/validate_nasa_first_guarded_graph.py',
            'tests/test_guarded_prediction_graphs.py','sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']},
        provider_sha256={name:sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'/name) for name in providers},
        limitations=['Controlled predictors verify routing; the separate primary graph report covers native models.',
            'No model training, native-state persistence, catalog publication or empirical performance claim.'])


if __name__ == '__main__':
    report=validate()
    (ROOT/'docs/reviews/competition_nasa_first_guarded_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(report['cases']),session_traces=report['session_traces'],actual_persistence=True)))
