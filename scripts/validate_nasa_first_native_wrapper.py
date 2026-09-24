"""Serial 21-fit source workflow from corrected synthetic raw inputs."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import warnings
import catboost
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sciona.prediction_postprocessing import clip_truncate_prediction
from scripts.synthetic_nasa_first_workflow_inputs import population_inputs
from sciona.nasa_first_prediction import nasa_first_prediction
from scripts.validate_nasa_first_native_training import SOURCE as TRAIN_SOURCE, PIN as TRAIN_PIN
from scripts.validate_nasa_first_global_population import BASE, PINS
from scripts.validate_nasa_first_prediction_arithmetic import SOURCE as PRED_SOURCE, PIN as PRED_PIN
from scripts.audit_nasa_first_training_closure import audit as audit_topology

ROOT=Path(__file__).resolve().parents[1]


def local_source_closure():
    pending=[Path(__file__).resolve()];seen={}
    while pending:
        path=pending.pop()
        name=str(path.relative_to(ROOT))
        if name in seen:continue
        data=path.read_bytes();seen[name]=hashlib.sha256(data).hexdigest()
        for node in ast.walk(ast.parse(data)):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith(('sciona.','scripts.')):
                dependency=ROOT.joinpath(*node.module.split('.')).with_suffix('.py')
                if dependency.is_file():pending.append(dependency)
    return seen


def source_function(source,path,digest,name):
    data=(source/path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Pinned software changed')
    tree=ast.parse(data)
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def validate(source, checkpoint_directory):
    checkpoint_directory.mkdir(parents=True,exist_ok=True)
    if any(checkpoint_directory.iterdir()):raise ValueError('Use an empty checkpoint directory to preserve previous runs')
    source_closure=local_source_closure()
    (checkpoint_directory/'source_closure.json').write_text(json.dumps(source_closure,indent=2)+'\n')
    topology=audit_topology(source)
    if topology['training_fit_nodes']!=21 or topology['residual_alias_pairs']!=10:raise ValueError('Topology differs')
    residual=source_function(source,TRAIN_SOURCE,TRAIN_PIN,'train_catboost_diff')
    direct=source_function(source,TRAIN_SOURCE,TRAIN_PIN,'train_catboost')
    global_builder=source_function(source,BASE+'nodes.py',PINS['nodes.py'],'build_global_master')
    predictor=source_function(source,PRED_SOURCE,PRED_PIN,'generate_predictions')
    labels=ast.literal_eval(next(n.value for n in global_builder.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='airports'))
    select=next(n for n in ast.walk(global_builder) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.BinOp) and isinstance(n.slice.left,ast.List))
    base_columns=ast.literal_eval(select.slice.left)
    markers=[n.left.value for n in ast.walk(select.slice.right) if isinstance(n,ast.Compare) and isinstance(n.left,ast.Constant)]
    label_column=next(n.targets[0].slice.value for n in ast.walk(global_builder) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript))
    cutoff=next(n for n in ast.walk(residual) if isinstance(n,ast.Compare) and isinstance(n.ops[0],ast.Gt))
    time_column=cutoff.left.slice.value
    offset_column=next(n.slice.value for n in ast.walk(residual) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='x_train' and isinstance(n.slice,ast.Constant))
    result_columns=ast.literal_eval(predictor.body[-2].value.slice)
    override_labels=ast.literal_eval(next(n.test.comparators[0] for n in predictor.body if isinstance(n,ast.If)))
    fit_contracts=[]
    class BoundedRegressor(CatBoostRegressor):
        def __init__(self,**kwargs):
            if kwargs and (kwargs.get('thread_count')!=-1 or kwargs.get('n_estimators')!=20000):raise ValueError('Source training settings differ')
            kwargs.update(thread_count=1,allow_writing_files=False)
            super().__init__(**kwargs)
        def fit(self,x,y,**kwargs):
            if kwargs['early_stopping_rounds']!=60 or not kwargs['use_best_model']:raise ValueError('Source stopping differs')
            expected=[i for i,c in enumerate(x.columns) if '_cat_' in c]
            if kwargs['cat_features']!=expected or not expected:raise ValueError('Categorical feature contract differs')
            fit_contracts.append(dict(training_rows=len(x),validation_rows=len(kwargs['eval_set'][0]),categorical_features=len(expected)))
            return super().fit(x,y,**kwargs)
        def predict(self,*args,**kwargs):
            kwargs['thread_count']=1
            return super().predict(*args,**kwargs)
        def get_feature_importance(self,*args,**kwargs):
            kwargs['thread_count']=1
            return super().get_feature_importance(*args,**kwargs)
    namespace={'pd':pd,'CatBoostRegressor':BoundedRegressor}
    exec(compile(ast.Module(body=[residual,direct,global_builder,predictor],type_ignores=[]),'pinned_population_workflow','exec'),namespace)
    time_start=pd.Timestamp(cutoff.comparators[0].value)+pd.Timedelta(days=1)
    end_train=str(time_start+pd.Timedelta(minutes=128*15));tables=[];raw_reports=[];raw_inputs=[]
    for population in range(10):
        table,raw_report,raw_queries,raw_options,policy=population_inputs(population,time_start)
        raw_inputs.append((raw_queries,raw_options,policy))
        tables.append(table);raw_reports.append(raw_report)
        print(json.dumps(dict(prepared_population=population,**raw_report)),flush=True)
    models=[];fits=[]
    def fit(table,name,population):
        with contextlib.redirect_stdout(io.StringIO()):
            result=namespace[name](table.copy(),base_columns[2],end_train)
        model=result[0] if isinstance(result,tuple) else result
        if isinstance(result,tuple):
            if result[0] is not result[1] or not np.isfinite(result[2]['imp']).all():raise ValueError('Residual state/importance invalid')
        fits.append(dict(population=population,kind='residual' if isinstance(result,tuple) else 'direct',
            retained_trees=model.tree_count_,best_iteration=model.get_best_iteration(),**fit_contracts[-1]))
        model.save_model(str(checkpoint_directory/f'model_{len(fits)-1}.cbm'))
        fits[-1]['model_sha256']=hashlib.sha256((checkpoint_directory/f'model_{len(fits)-1}.cbm').read_bytes()).hexdigest()
        (checkpoint_directory/'fit_progress.json').write_text(json.dumps(fits,indent=2)+'\n')
        print(json.dumps(dict(completed_fits=len(fits),**fits[-1])),flush=True)
        return model
    for i,table in enumerate(tables):
        r=fit(table,'train_catboost_diff',i);d=fit(table,'train_catboost',i)
        models.append({0:r,1:d,2:r})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.SettingWithCopyWarning)
        global_table=namespace['build_global_master'](*[t.copy() for t in tables])
    global_model=fit(global_table,'train_catboost',10)
    compared=0;roundtrips=0;wrapper_compared=0;fallback_cases=0
    with tempfile.TemporaryDirectory(prefix='sciona-catboost-populations-') as directory:
        for i,table in enumerate(tables):
            model=models[i];model['global_model']=global_model
            query=table.iloc[np.arange(128,192)*128].copy();query[result_columns[2]]=labels[i]
            saved={}
            for slot,estimator in model.items():
                path=Path(directory)/f'population{i}_{slot}.cbm';estimator.save_model(str(path))
                restored=BoundedRegressor();restored.load_model(str(path));saved[slot]=restored
                features=query.copy();features[label_column]=labels[i]
                np.testing.assert_array_equal(estimator.predict(features[estimator.feature_names_]),restored.predict(features[restored.feature_names_]))
                roundtrips+=1
            actual=namespace['generate_predictions'](saved,query.copy(),labels[i])[result_columns[-1]].to_numpy()
            baseline=query[offset_column].to_numpy();local=[]
            for slot in range(3):
                raw=saved[slot].predict(query[saved[slot].feature_names_])
                local.append(clip_truncate_prediction(raw,baseline if slot in (0,2) else np.zeros(len(query)),1,299))
            query[label_column]=labels[i]
            raw_global=saved['global_model'].predict(query[saved['global_model'].feature_names_])
            raw=local[1] if labels[i].upper() in override_labels else (local[0]+local[1]+local[2]+raw_global)/4
            expected=clip_truncate_prediction(raw,np.zeros(len(query)),4,260)
            np.testing.assert_array_equal(actual,expected);compared+=len(actual)
            raw_queries,raw_options,policy=raw_inputs[i]
            held_out=raw_queries.iloc[128:].copy();held_out[result_columns[2]]=labels[i]
            wrapped,route=nasa_first_prediction(held_out,saved,labels[i],raw_options=raw_options,**policy)
            if route!='primary':raise ValueError('Native wrapper unexpectedly fell back')
            np.testing.assert_array_equal(wrapped[result_columns[-1]],expected);wrapper_compared+=len(wrapped)
            repeated,route=nasa_first_prediction(held_out.iloc[[63,0,63]],saved,labels[i],raw_options=raw_options,**policy)
            if route!='primary':raise ValueError('Repeated-query native route differs')
            pd.testing.assert_frame_equal(repeated,wrapped.iloc[[63,0,63]])
            baseline,route=nasa_first_prediction(held_out,{},labels[i],raw_options=raw_options,**policy)
            if route!='baseline':raise ValueError('Missing models did not select baseline')
            expected_baseline=np.clip(query[offset_column].to_numpy()-30,1,299)
            np.testing.assert_array_equal(baseline[result_columns[-1]],expected_baseline);fallback_cases+=1
            constant,route=nasa_first_prediction(held_out,{},labels[i],raw_options={},**policy)
            if route!='constant':raise ValueError('Double failure did not select constant')
            np.testing.assert_array_equal(constant[result_columns[-1]],30);fallback_cases+=1

    if len(fits)!=21 or len({id(v) for m in models for v in m.values()})!=21:raise ValueError('Independent fit identity count differs')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,backend_version=catboost.__version__,numpy_version=np.__version__,pandas_version=pd.__version__,native_threads=1,
        source_sha256={TRAIN_SOURCE:TRAIN_PIN,BASE+'nodes.py':PINS['nodes.py'],PRED_SOURCE:PRED_PIN},
        local_source_closure_sha256=source_closure,
        raw_population_reports=raw_reports,fits=fits,training_fit_count=21,model_output_slots=31,residual_alias_pairs=10,global_model_shared=True,
        serialization_prediction_checks=roundtrips,exact_submission_predictions=compared,
        native_wrapper_distinct_query_comparisons=wrapper_compared,native_wrapper_fallback_cases=fallback_cases,
        native_wrapper_duplicate_order_cases=10,
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/validate_nasa_first_native_wrapper.py','scripts/synthetic_nasa_first_workflow_inputs.py','sciona/nasa_first_prediction.py','sciona/asof_estimate_fallback.py','sciona/prediction_fallback_chain.py','sciona/nasa_first_feature_adapter.py','sciona/model_feature_policy.py','sciona/prediction_postprocessing.py','scripts/audit_nasa_first_training_closure.py','scripts/validate_nasa_first_native_training.py','scripts/validate_nasa_first_global_population.py','scripts/validate_nasa_first_prediction_arithmetic.py']},
        limitations=['Synthetic raw inputs with explicit repeated queries exercise all seven corrected feature families; this is workflow qualification, not competition-performance evidence. Primary and injected fallback routes are checked through the final raw-input wrapper.',
            'Current CatBoost backend and provisioned in-process execution; no historical 1.1.1 parity or competition-performance claim.',
            'Explicit corrected training-to-submission composition; historical training prediction-helper unresolved names remain recorded.',
            'No CDG assembly, provider registration or publication approval applied.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True)
    args=parser.parse_args()
    report=validate(args.source_directory,args.checkpoint_directory)
    (args.checkpoint_directory/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'docs/reviews/competition_nasa_first_native_wrapper.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,fits=21,exact_submission_predictions=report['exact_submission_predictions'],approved=False)))
