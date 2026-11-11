"""Run all source-sized synthetic fits through the complete production training graph."""
import argparse
import ast
import asyncio
import contextlib
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sciona.nasa_first_population_training_graph import build_nasa_first_population_training_graph
from sciona.nasa_first_prediction import nasa_first_prediction
from sciona.ghost.registry import REGISTRY
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.synthetic_nasa_first_workflow_inputs import population_inputs
from scripts.validate_nasa_first_native_wrapper import source_function,TRAIN_SOURCE,TRAIN_PIN
from scripts.validate_nasa_first_model_handoff import ROOT,sha


def closure(graph):
    roots={'sciona-matcher':ROOT,'sciona-atoms-ml':ROOT.parent/'sciona-atoms-ml/src',
           'sciona-atoms':ROOT.parent/'sciona-atoms/src'}
    pending=[Path(__file__),ROOT/'sciona/visualizer/runner.py',ROOT/'sciona/services/execution_graph_codec.py']
    pending += [Path(inspect.getsourcefile(REGISTRY[node.matched_primitive]['impl'])) for node in graph.nodes]
    seen={}
    while pending:
        path=pending.pop().resolve()
        matched=next(((name,base) for name,base in roots.items() if path.is_relative_to(base)),None)
        if matched is None:continue
        label,base=matched;key=label+'/'+str(path.relative_to(base))
        if key in seen:continue
        seen[key]=sha(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            modules=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [n.name for n in node.names] if isinstance(node,ast.Import) else []
            for module in modules:
                if not module.startswith(('sciona.','scripts.')):continue
                for base in roots.values():
                    candidate=base.joinpath(*module.split('.')).with_suffix('.py')
                    if candidate.is_file():pending.append(candidate)
    return dict(sorted(seen.items()))


def validate(source,checkpoint,reference):
    checkpoint.mkdir(parents=True,exist_ok=True)
    if any(checkpoint.iterdir()):raise ValueError('Empty checkpoint directory required; prior runs must remain intact')
    runner._ensure_atoms_imported()
    from sciona.atoms.ml.domain_adapters.first_place_populations import POPULATIONS
    graph=build_nasa_first_population_training_graph()
    digest,nodes,edges=encode_execution_graph(graph);restored=decode_execution_graph(nodes,edges,digest)
    if restored!=graph:raise ValueError('Training graph codec differs')
    frozen=closure(graph)
    (checkpoint/'source_closure.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (checkpoint/'graph.json').write_text(graph.model_dump_json(indent=2))
    prior=json.loads((ROOT/'docs/reviews/competition_nasa_first_native_wrapper.json').read_text())
    if not prior['passed'] or json.loads((reference/'qualification.json').read_text())!=prior:
        raise ValueError('Qualified reference models required')
    for index,fit in enumerate(prior['fits']):
        if sha(reference/f'model_{index}.cbm')!=fit['model_sha256']:raise ValueError('Reference model integrity differs')
    residual=source_function(source,TRAIN_SOURCE,TRAIN_PIN,'train_catboost_diff')
    cutoff=next(n for n in ast.walk(residual) if isinstance(n,ast.Compare) and isinstance(n.ops[0],ast.Gt))
    start=pd.Timestamp(cutoff.comparators[0].value)+pd.Timedelta(days=1)
    end_train=str(start+pd.Timedelta(minutes=128*15))
    tables={};queries=[];raw_options=[];policies=[];raw_reports=[]
    for index,label in enumerate(POPULATIONS):
        table,report,query,raw,policy=population_inputs(index,start)
        tables[label]=table;queries.append(query);raw_options.append(raw);policies.append(policy);raw_reports.append(report)
        print(json.dumps(dict(prepared_population=index,**report)),flush=True)
    progress=[];fit_inputs={};captured={};stream=sys.stdout
    def record(path,node,port,value):
        if node.endswith('/fit'):
            observed=fit_inputs.setdefault(node,{})
            if port in ('in_x_train','in_x_valid'):observed[port[3:]+'_rows']=len(value)
            if port=='in_categorical_indices':observed['categorical_features']=len(value)
            if port=='out_model':
                group=node.split('/')[0]
                index=2*int(group.removeprefix('residual')) if group.startswith('residual') else 20 if group=='direct10' else 2*int(group.removeprefix('direct'))+1
                params=value.get_params()
                if params['thread_count']!=1 or params['allow_writing_files'] is not False or params['n_estimators']!=20000:
                    raise ValueError('Native fit execution policy differs')
                model_path=checkpoint/f'model_{index}.cbm';value.save_model(str(model_path))
                item=dict(model_index=index,node_id=node,retained_trees=value.tree_count_,best_iteration=value.get_best_iteration(),
                    model_sha256=sha(model_path),**observed)
                progress.append(item)
                temporary=checkpoint/'fit_progress.next.json';temporary.write_text(json.dumps(progress,indent=2)+'\n')
                temporary.replace(checkpoint/'fit_progress.json')
                print(json.dumps(dict(completed_fits=len(progress),model_index=index,retained_trees=value.tree_count_)),file=stream,flush=True)
        if node=='bank' and port=='out_bank':captured['bank']=value
        if node.endswith('/residual_outputs') and port=='out_feature_importance':
            if not np.isfinite(value['imp']).all():raise ValueError('Nonfinite normalized importance')
            captured.setdefault('importance',[]).append(node)
    with tempfile.TemporaryDirectory(prefix='sciona-native-training-graph-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=record), \
                contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-native-training','qualification').execute(
                dict(population_tables=tables,target_name='minutes_until_pushback',end_train=end_train),cdg=restored))
    if result['status']!='completed' or len(result['trace'])!=202 or len(progress)!=21 or len(captured.get('importance',[]))!=10:
        raise ValueError('Full native graph execution incomplete')
    bank=captured['bank'];compared=0;model_compared=0
    global_reference=CatBoostRegressor(thread_count=1);global_reference.load_model(str(reference/'model_20.cbm'))
    for index,label in enumerate(POPULATIONS):
        slots=bank[label.upper()]
        if slots[0] is not slots[2] or slots['global_model'] is not bank[POPULATIONS[0].upper()]['global_model']:
            raise ValueError('Native model-bank alias differs')
        reference_slots={'global_model':global_reference}
        for slot,number in [(0,2*index),(1,2*index+1)]:
            model=CatBoostRegressor(thread_count=1);model.load_model(str(reference/f'model_{number}.cbm'));reference_slots[slot]=model
        reference_slots[2]=reference_slots[0]
        query=queries[index].iloc[128:]
        expected,route=nasa_first_prediction(query,reference_slots,label,raw_options=raw_options[index],**policies[index])
        actual,actual_route=nasa_first_prediction(query,slots,label,raw_options=raw_options[index],**policies[index])
        if route!=actual_route or route!='primary':raise ValueError('Graph-trained native model fell back')
        pd.testing.assert_frame_equal(actual,expected);compared+=len(query)
        features=tables[label].iloc[np.arange(128,192)*128].copy();features['feat_cat_airport']=label
        for slot,model in slots.items():
            np.testing.assert_array_equal(model.predict(features[model.feature_names_],thread_count=1),
                reference_slots[slot].predict(features[reference_slots[slot].feature_names_],thread_count=1))
            model_compared+=len(features)
    if closure(graph)!=frozen:raise ValueError('Source changed during native qualification')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,native_threads=1,
        graph_sha256=digest,nodes=202,edges=646,executed_nodes=len(result['trace']),fits=progress,independent_native_fits=21,
        residual_importance_tables=10,residual_alias_pairs=10,shared_global_model=True,
        exact_native_prediction_comparisons=model_compared,exact_final_prediction_comparisons=compared,
        raw_population_reports=raw_reports,source_closure_sha256=frozen,
        reference_qualification_sha256=sha(ROOT/'docs/reviews/competition_nasa_first_native_wrapper.json'),
        limitations=['Prepared training tables originate from corrected synthetic raw features; the raw extraction step precedes this training graph.',
            'Repeated synthetic queries exercise the source large-leaf setting; this is computational qualification, not predictive performance evidence.',
            'Intermediate capture writes model checkpoints instead of standard visualization persistence; all computations and fit dispatch use production implementations.',
            'Catalog binding/publication and portable native-state envelopes remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True);parser.add_argument('--reference-checkpoints',type=Path,required=True)
    args=parser.parse_args();report=validate(args.source_directory,args.checkpoint_directory,args.reference_checkpoints)
    (args.checkpoint_directory/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'docs/reviews/competition_nasa_first_native_training_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,native_fits=21,exact_final_predictions=report['exact_final_prediction_comparisons'])))
