"""Prove raw graph matrices equal those used by the qualified native training run."""
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
from sciona.nasa_first_raw_training_graph import build_nasa_first_raw_training_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.synthetic_nasa_first_workflow_inputs import population_inputs
from scripts.validate_nasa_first_native_wrapper import source_function,TRAIN_SOURCE,TRAIN_PIN
from scripts.validate_nasa_first_model_handoff import ROOT,sha


def validate(source):
    qualification_path=ROOT/'docs/reviews/competition_nasa_first_native_training_graph.json'
    qualification=json.loads(qualification_path.read_text())
    if not qualification['passed'] or qualification['independent_native_fits']!=21:
        raise ValueError('Native population training qualification required')
    roots={'sciona-matcher':ROOT,'sciona-atoms-ml':ROOT.parent/'sciona-atoms-ml/src','sciona-atoms':ROOT.parent/'sciona-atoms/src'}
    for name,digest in qualification['source_closure_sha256'].items():
        repository,relative=name.split('/',1)
        if sha(roots[repository]/relative)!=digest:raise ValueError('Native training evidence source drift')
    residual=source_function(source,TRAIN_SOURCE,TRAIN_PIN,'train_catboost_diff')
    cutoff=next(node for node in ast.walk(residual) if isinstance(node,ast.Compare) and isinstance(node.ops[0],ast.Gt))
    start=pd.Timestamp(cutoff.comparators[0].value)+pd.Timedelta(days=1)
    from sciona.atoms.ml.domain_adapters.first_place_populations import POPULATIONS
    raw={};expected={}
    for index,label in enumerate(POPULATIONS):
        table,report,queries,options,policy=population_inputs(index,start)
        if report!=qualification['raw_population_reports'][index]:raise ValueError('Qualified synthetic population differs')
        training_queries=table[['gufi','timestamp','minutes_until_pushback']].copy()
        raw[label]=dict(queries=training_queries,raw_options=options,**policy)
        expected[label]=table
    graph=build_nasa_first_raw_training_graph()
    digest,nodes,edges=encode_execution_graph(graph);restored=decode_execution_graph(nodes,edges,digest)
    if restored!=graph:raise ValueError('Raw training graph codec differs')
    captured={}
    def capture(path,node,port,value):
        if node=='collect_tables' and port=='out_population_tables':captured['tables']=value
    with tempfile.TemporaryDirectory(prefix='sciona-raw-training-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture), \
                contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-raw-training','qualification').execute(
                dict(raw_populations=raw,target_name='minutes_until_pushback',end_train=str(start+pd.Timedelta(minutes=128*15))),
                target_node_id='collect_tables',cdg=restored))
    if result['status']!='completed' or len(result['trace'])!=132 or any(item['node_id'].endswith('/fit') for item in result['trace']):
        raise ValueError('Raw preparation execution scope differs')
    if set(captured['tables'])!=set(expected):raise ValueError('Prepared population coverage differs')
    rows=0
    for label,table in expected.items():
        pd.testing.assert_frame_equal(captured['tables'][label],table,check_exact=True)
        rows+=len(table)
        pd.testing.assert_frame_equal(raw[label]['queries'],table[['gufi','timestamp','minutes_until_pushback']],check_exact=True)
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        full_graph_nodes=len(graph.nodes),full_graph_edges=len(graph.edges),graph_sha256=digest,
        executed_preparation_nodes=132,populations=10,exact_prepared_rows=rows,model_features_per_population=260,
        matched_native_training_qualification_sha256=sha(qualification_path),
        native_fit_execution_in_this_run=False,
        implementation_sha256={name:sha(ROOT/name) for name in ['sciona/nasa_first_raw_training_graph.py',
            'scripts/validate_nasa_first_raw_training_graph.py','scripts/synthetic_nasa_first_workflow_inputs.py',
            'sciona/nasa_first_feature_graph.py','sciona/nasa_first_population_training_graph.py']},
        provider_sha256=sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/first_place_raw_training.py'),
        limitations=['Raw preparation and native population training are separately executed and joined by exact full-frame equality; no second training run is claimed.',
            'Synthetic repeated rows qualify computation and source large-leaf behavior, not empirical predictive quality.',
            'Model/policy state binding and catalog publication remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_raw_training_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,nodes=report['full_graph_nodes'],edges=report['full_graph_edges'],exact_prepared_rows=report['exact_prepared_rows'])))
