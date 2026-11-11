"""Verify complete lifecycle dataflow with exact qualified fit-operand replay.

Only the 21 expensive fit calls are replayed from qualified native checkpoints,
after comparing every fit operand. Raw preparation, scoring, state serialization
and inference execute normally. This is compositional evidence, not a new fit.
"""
import argparse
import ast
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from catboost import CatBoostRegressor
import pandas as pd
from sciona.architect.handoff import CDGExport
from sciona.ghost.registry import REGISTRY
from sciona.nasa_first_complete_graph import build_nasa_first_complete_graph,root_contracts
from sciona.nasa_first_state_graphs import build_nasa_first_training_state_graph,build_nasa_first_state_inference_graph
from sciona.nasa_first_prediction import nasa_first_prediction
from sciona.named_regression_training import temporal_regression_split
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.synthetic_nasa_first_workflow_inputs import population_inputs
from scripts.validate_nasa_first_native_wrapper import source_function,TRAIN_SOURCE,TRAIN_PIN
from scripts.validate_nasa_first_native_training_graph import closure
from scripts.validate_nasa_first_model_handoff import ROOT,sha


def validate(source,checkpoint):
    runner._ensure_atoms_imported()
    from sciona.atoms.ml.domain_adapters.first_place_populations import POPULATIONS,training_tables
    from sciona.atoms.ml.domain_adapters.first_place_training import residual_inputs,direct_inputs
    from sciona.population_tables import concatenate_labeled_populations
    graph=build_nasa_first_complete_graph();root_ports=root_contracts(graph)
    for prefix,original in [('training/',build_nasa_first_training_state_graph()),('inference/',build_nasa_first_state_inference_graph())]:
        nodes=[n.model_copy(deep=True,update={'node_id':n.node_id.removeprefix(prefix)}) for n in graph.nodes if n.node_id.startswith(prefix)]
        edges=[e.model_copy(deep=True,update={'source_id':e.source_id.removeprefix(prefix),'target_id':e.target_id.removeprefix(prefix)})
               for e in graph.edges if e.source_id.startswith(prefix) and e.target_id.startswith(prefix)]
        if CDGExport(nodes=nodes,edges=edges,metadata=original.metadata)!=original:
            raise ValueError('Combined section differs from qualified graph')
    cross=[e for e in graph.edges if e.source_id.split('/')[0]!=e.target_id.split('/')[0]]
    if len(cross)!=1 or cross[0].source_id!='training/bind_state' or cross[0].target_id!='inference/unbind_state':
        raise ValueError('Explicit single state handoff required')
    digest,nodes,edges=encode_execution_graph(graph)
    if decode_execution_graph(nodes,edges,digest)!=graph:raise ValueError('Complete graph codec differs')
    native_path=ROOT/'docs/reviews/competition_nasa_first_native_training_graph.json'
    native=json.loads(native_path.read_text())
    if not native['passed'] or json.loads((checkpoint/'qualification.json').read_text())!=native:
        raise ValueError('Qualified checkpoint report required')
    roots={'sciona-matcher':ROOT,'sciona-atoms-ml':ROOT.parent/'sciona-atoms-ml/src','sciona-atoms':ROOT.parent/'sciona-atoms/src'}
    for name,value in native['source_closure_sha256'].items():
        repo,relative=name.split('/',1)
        if sha(roots[repo]/relative)!=value:raise ValueError('Qualified native source changed')
    frozen=closure(graph)
    models={}
    for fit in native['fits']:
        index=fit['model_index'];path=checkpoint/f'model_{index}.cbm'
        if sha(path)!=fit['model_sha256']:raise ValueError('Qualified model changed')
        model=CatBoostRegressor(thread_count=1);model.load_model(str(path));models[index]=model
    residual=source_function(source,TRAIN_SOURCE,TRAIN_PIN,'train_catboost_diff')
    cutoff=next(n for n in ast.walk(residual) if isinstance(n,ast.Compare) and isinstance(n.ops[0],ast.Gt))
    start=pd.Timestamp(cutoff.comparators[0].value)+pd.Timedelta(days=1)
    end=str(start+pd.Timedelta(minutes=128*15));raw={};tables={};heldout={}
    for index,label in enumerate(POPULATIONS):
        table,report,queries,options,policy=population_inputs(index,start)
        if report!=native['raw_population_reports'][index]:raise ValueError('Qualified training population differs')
        tables[label]=table
        raw[label]=dict(queries=table[['gufi','timestamp','minutes_until_pushback']].copy(),raw_options=options,**policy)
        heldout[label]=(queries.iloc[128:],options,policy)
    values=training_tables(tables,'minutes_until_pushback')
    global_table=concatenate_labeled_populations(*values[-5:])
    expected={}
    for index,label in enumerate(POPULATIONS):
        expected[f'training/residual{index}/fit']=(2*index,residual_inputs(tables[label],'minutes_until_pushback',end))
        expected[f'training/direct{index}/fit']=(2*index+1,direct_inputs(tables[label],'minutes_until_pushback',end))
    expected['training/direct10/fit']=(20,direct_inputs(global_table,'minutes_until_pushback',end))
    for node,(index,operands) in list(expected.items()):
        frame,targets,times,eligible,boundary,offsets,categories,parameters,patience=operands
        expected[node]=(index,(*temporal_regression_split(frame,targets,times,eligible,boundary,offsets,categories),parameters,patience))
    runtime='sciona.atoms.ml.model_selection.named_regression.fit'
    active=[None];replayed=[];captured={}
    def replay(x_train,y_train,x_valid,y_valid,categorical_indices,parameters,early_stopping_rounds):
        actual=(x_train,y_train,x_valid,y_valid,categorical_indices,parameters,early_stopping_rounds)
        node=active[0]
        if node not in expected or node in replayed:raise ValueError('Unexpected fit replay')
        index,wanted=expected[node]
        for left,right in zip(actual[:4],wanted[:4]):
            if isinstance(right,pd.DataFrame):pd.testing.assert_frame_equal(left,right,check_exact=True)
            else:pd.testing.assert_series_equal(left,right,check_exact=True)
        if actual[4:]!=wanted[4:]:raise ValueError('Native fit policy or categories differ')
        replayed.append(node);return models[index]
    def capture(path,node,port,value):
        if node.endswith('/fit') and port.startswith('in_'):active[0]=node
        if (node,port) in [('training/bind_state','out_state'),('inference/format','out_predictions')]:captured[node]=value
    label=POPULATIONS[0];query,options,policy=heldout[label]
    with tempfile.TemporaryDirectory(prefix='sciona-complete-lifecycle-') as directory:
        with patch.dict(REGISTRY,{runtime:dict(REGISTRY[runtime],impl=replay)}),patch.object(runner,'RUNS_DIR',Path(directory)),\
                patch.object(runner,'save_intermediate_value',side_effect=capture),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-complete-lifecycle','qualification').execute(
                dict(raw_populations=raw,target_name='minutes_until_pushback',end_train=end,population=label,queries=query,
                     raw_options={k:v for k,v in options.items() if k not in ['vocabularies','holiday_midnights']}),cdg=graph))
    if result['status']!='completed' or len(result['trace'])!=343 or len(replayed)!=21:raise ValueError('Complete lifecycle traversal differs')
    slots={0:models[0],1:models[1],2:models[0],'global_model':models[20]}
    predictions,route=nasa_first_prediction(query,slots,label,raw_options=options,**policy)
    if route!='primary':raise ValueError('Reference prediction fallback')
    pd.testing.assert_frame_equal(captured['inference/format'],predictions,check_exact=True)
    json.dumps(captured['training/bind_state'],allow_nan=False)
    if closure(graph)!=frozen:raise ValueError('Implementation changed during lifecycle execution')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,native_threads=1,
        graph_sha256=digest,nodes=343,edges=len(graph.edges),executed_outer_nodes=len(result['trace']),
        exact_replayed_fit_operands=21,new_native_fits=0,exact_native_prediction_comparisons=len(query),
        portable_state_produced=True,qualified_section_projection_equal=True,root_inputs=root_ports,
        native_qualification_sha256=sha(native_path),source_closure_sha256=frozen,
        implementation_sha256={name:sha(ROOT/name) for name in ['sciona/nasa_first_complete_graph.py','scripts/validate_nasa_first_complete_graph.py']},
        limitations=['Compositional qualification: all fit operands match independently rebuilt qualified matrices and policies before checkpoint replay.',
                     'The original 21 native fits are bound by checkpoint hashes and source closure; no new native fitting is claimed.',
                     'All remaining computation and state handoff run through the production executor; intermediate persistence uses capture hooks.',
                     'Stored original-CDG retrieval and publication remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True);args=parser.parse_args()
    report=validate(args.source_directory,args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_complete_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','nodes','edges','exact_replayed_fit_operands','exact_native_prediction_comparisons']}))
