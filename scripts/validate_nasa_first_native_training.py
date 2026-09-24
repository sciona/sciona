"""Run pinned direct/residual training on synthetic data with bounded native threads."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import catboost
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/training/src/pushback_nasa/pipelines/train_models/nodes.py'
PIN='9b96dd580d39b9ae6126c380c390f57c21b10d3cbc7d028af5a6639799da271d'


def validate(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned trainer changed')
    tree=ast.parse(data)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    residual=next(n for n in functions if n.name=='train_catboost_diff')
    cutoff=next(n for n in ast.walk(residual) if isinstance(n,ast.Compare) and isinstance(n.ops[0],ast.Gt))
    time_field=cutoff.left.slice.value
    excluded=[n.args[0].value for n in ast.walk(residual) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='remove' and isinstance(n.args[0],ast.Constant)]
    identity_field=next(n for n in excluded if n!=time_field)
    offset_field=next(n.slice.value for n in ast.walk(residual) if isinstance(n,ast.Subscript)
        and isinstance(n.value,ast.Name) and n.value.id=='x_train' and isinstance(n.slice,ast.Constant))
    rng=np.random.default_rng(884)
    count=24576;training_count=16384
    time_start=pd.Timestamp(cutoff.comparators[0].value)+pd.Timedelta(days=1)
    times=pd.date_range(time_start,periods=count,freq='min')
    end_train=str(times[training_count])
    numbers=rng.normal(size=count)
    frame=pd.DataFrame({identity_field:np.arange(count),time_field:times,'synthetic_target':100+rng.uniform(1,100,size=count),
        offset_field:80+numbers,'synthetic_feature':rng.normal(size=count)})
    observed=[]
    class BoundedRegressor(CatBoostRegressor):
        def __init__(self,**kwargs):
            if kwargs.get('thread_count')!=-1 or kwargs.get('n_estimators')!=20000:raise ValueError('Source resource/schedule assumptions differ')
            kwargs.update(thread_count=1,allow_writing_files=False)
            super().__init__(**kwargs)
        def fit(self,x,y,**kwargs):
            if kwargs.get('early_stopping_rounds')!=60 or not kwargs.get('use_best_model'):raise ValueError('Source stopping differs')
            expected=frame['synthetic_target'].to_numpy()
            if len(observed)==0:expected=expected-frame[offset_field].to_numpy()
            np.testing.assert_array_equal(np.asarray(y),expected[:training_count])
            np.testing.assert_array_equal(np.asarray(kwargs['eval_set'][1]),expected[training_count:])
            if len(x)!=training_count or len(kwargs['eval_set'][0])!=count-training_count:raise ValueError('Split differs')
            observed.append(True)
            return super().fit(x,y,**kwargs)
        def predict(self,*args,**kwargs):
            kwargs['thread_count']=1
            return super().predict(*args,**kwargs)
        def get_feature_importance(self,*args,**kwargs):
            kwargs['thread_count']=1
            return super().get_feature_importance(*args,**kwargs)
    namespace={'pd':pd,'CatBoostRegressor':BoundedRegressor}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'pinned_trainers','exec'),namespace)
    reports=[]
    for name in ['train_catboost_diff','train_catboost']:
        with contextlib.redirect_stdout(io.StringIO()):
            result=namespace[name](frame.copy(),'synthetic_target',end_train)
        model=result[0] if isinstance(result,tuple) else result
        if isinstance(result,tuple) and result[0] is not result[1]:raise ValueError('Residual state alias lost')
        features=frame.iloc[training_count:][model.feature_names_]
        predictions=model.predict(features)
        if not np.isfinite(predictions).all():raise ValueError('Nonfinite native prediction')
        with tempfile.TemporaryDirectory(prefix='sciona-catboost-synthetic-') as directory:
            path=Path(directory)/'model.cbm';model.save_model(str(path))
            restored=CatBoostRegressor();restored.load_model(str(path))
            np.testing.assert_array_equal(restored.predict(features,thread_count=1),predictions)
        reports.append(dict(trainer=name,training_rows=training_count,validation_rows=count-training_count,
            retained_trees=model.tree_count_,best_iteration=model.get_best_iteration(),
            residual_state_alias=isinstance(result,tuple),prediction_roundtrip_exact=True,
            feature_importance_finite=bool(np.isfinite(result[2]['imp']).all()) if isinstance(result,tuple) else None))
        print(json.dumps(reports[-1]),flush=True)
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=PIN,
        backend_version=catboost.__version__,numpy_version=np.__version__,pandas_version=pd.__version__,
        native_threads=1,maximum_trees=20000,early_stopping_rounds=60,fits=reports,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Two isolated native source trainers on synthetic numeric features; not the full 21-fit workflow or real competition data.',
            'Source CatBoost 1.1.1 is not reproduced; this qualifies the current provisioned backend only.',
            'Source constructor thread count and prediction/importance threads explicitly bounded to one; native file logging disabled.',
            'Feature extraction, categorical coverage, global population construction, routing and catalog publication remain separate gates.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_native_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,fits=len(report['fits']),approved=False)))
