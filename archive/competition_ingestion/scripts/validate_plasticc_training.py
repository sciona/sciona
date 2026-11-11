"""Actual synthetic LightGBM folds, original source parity and native API oracle."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import plasticc_training as runtime
from sciona.plasticc_weighting import label_folds


class Population:
    def __init__(self, metadata, features):
        self.metadata=metadata; self.raw_features=features; self.name='synthetic'
    def select_features(self, featurizer):
        return featurizer.select_features(self.raw_features)
    def label_folds(self,num_folds=None,random_state=None):
        return label_folds(self,num_folds,random_state,settings=dict(num_folds=3,fold_random_state=4))


class Identity:
    def select_features(self,raw): return raw.copy()


def validate(root):
    path=root/'avocado/classifier.py'
    pins=json.loads((ROOT/'docs/reviews/competition_plasticc_source_pins.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['avocado/classifier.py']
    ns=dict(vars(runtime))
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LightGBMClassifier'
        or isinstance(n,ast.FunctionDef) and n.name=='fit_lightgbm_classifier']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    captured=[]
    # Execute original old-API source using a narrow compatibility boundary.
    class Legacy:
        def __init__(self,**kwargs): self.impl=lgb.LGBMClassifier(**kwargs); captured.append(kwargs)
        def fit(self,x,y,verbose,early_stopping_rounds,**kwargs):
            return self.impl.fit(x,y,callbacks=[lgb.log_evaluation(verbose),lgb.early_stopping(early_stopping_rounds)],**kwargs)
        def __getattr__(self,name): return getattr(self.impl,name)
    legacy_module=SimpleNamespace(LGBMClassifier=Legacy)
    rng=np.random.default_rng(74)
    counters=dict(trained_folds=0,source_model_parity=0,out_of_fold_predictions=0,
                  prediction_ensemble=0,native_api_oracles=0,early_stopped=0,default_parameters=0)
    for scenario in range(2):
        labels=np.repeat([6,42,90],60)
        values=rng.normal(size=(180,5))
        if scenario==0: values[:,0]+=np.repeat([0,3,6],60)
        index=['synthetic-%d'%i for i in range(180)]
        features=pd.DataFrame(values,index=index,columns=['x%d'%i for i in range(5)])
        metadata=pd.DataFrame({'class':labels},index=index)
        # Include augmentations and validate actual held-out predictions with original grouping.
        aug=metadata.iloc[::5].copy(); aug['reference_object_id']=aug.index; aug.index=aug.index+'-aug'
        aug_features=features.iloc[::5].copy(); aug_features.index=aug.index
        population=Population(pd.concat([metadata,aug]),pd.concat([features,aug_features]))
        config=dict(n_estimators=100,n_jobs=1,random_state=7,random_seed=7,min_child_weight=.01,min_split_gain=0.,min_child_samples=5)
        actual=runtime.LightGBMClassifier('synthetic',Identity(),class_weights={6:1,42:2,90:1})
        original=ns['LightGBMClassifier']('synthetic',Identity(),class_weights={6:1,42:2,90:1})
        with contextlib.redirect_stdout(io.StringIO()):
            actual.train(population,**config)
            with patch.dict(sys.modules,lightgbm=legacy_module): original.train(population,**config)
        pd.testing.assert_frame_equal(actual.train_predictions,original.train_predictions)
        pd.testing.assert_frame_equal(actual.importances,original.importances)
        folds=population.label_folds(random_state=7)
        for i,(a,b) in enumerate(zip(actual.classifiers,original.classifiers)):
            assert a.booster_.dump_model()==b.impl.booster_.dump_model()
            assert a.booster_.num_trees() >= 3
            if scenario == 0: assert a.booster_.num_trees() > 3
            mask=folds==i
            np.testing.assert_allclose(actual.train_predictions[mask],a.predict_proba(population.raw_features[mask],num_iteration=a.best_iteration_))
            counters['trained_folds']+=1; counters['source_model_parity']+=1; counters['out_of_fold_predictions']+=1
            counters['early_stopped']+=int(a.best_iteration_<config['n_estimators'])
        prediction=actual.predict(population)
        pd.testing.assert_frame_equal(prediction,original.predict(population))
        expected=np.mean([c.predict_proba(population.raw_features,num_iteration=c.best_iteration_) for c in actual.classifiers],axis=0)
        np.testing.assert_allclose(prediction,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(prediction.sum(axis=1),1)
        counters['prediction_ensemble']+=1
        # Independent LightGBM native Dataset/train API for the first fold.
        train=folds!=0; valid=~train
        weights=actual.weighting_function(population,actual.class_weights)
        encoded=population.metadata['class'].map({6:0,42:1,90:2})
        params=dict(boosting_type='gbdt',objective='multiclass',num_class=3,metric='multi_logloss',
            learning_rate=.05,feature_fraction=.5,lambda_l1=0.,lambda_l2=0.,min_gain_to_split=0.,
            min_sum_hessian_in_leaf=.01,max_depth=7,num_leaves=50,min_data_in_leaf=5,num_threads=1,
            seed=7,verbosity=-1)
        train_set=lgb.Dataset(population.raw_features[train],label=encoded[train],weight=weights[train])
        valid_set=lgb.Dataset(population.raw_features[valid],label=encoded[valid],weight=weights[valid],reference=train_set)
        native=lgb.train(params,train_set,num_boost_round=100,valid_sets=[valid_set],callbacks=[lgb.early_stopping(50,verbose=False)])
        np.testing.assert_allclose(native.predict(population.raw_features[valid]),actual.train_predictions[valid],rtol=1e-12,atol=1e-12)
        counters['native_api_oracles']+=1
    assert counters['early_stopped']>0
    # Capture unchanged source defaults without claiming a competition-length run.
    defaults=[]
    class Capture:
        def __init__(self,**kwargs): defaults.append(kwargs)
        def fit(self,*args,**kwargs): pass
    fake=SimpleNamespace(LGBMClassifier=Capture,log_evaluation=lgb.log_evaluation,early_stopping=lgb.early_stopping)
    with patch.dict(sys.modules,lightgbm=fake):
        runtime.fit_lightgbm_classifier(features,metadata['class'],np.ones(180),features,metadata['class'],np.ones(180))
        ns['fit_lightgbm_classifier'](features,metadata['class'],np.ones(180),features,metadata['class'],np.ones(180))
    assert defaults[0]==defaults[1]
    assert defaults[0]['n_estimators']==5000 and defaults[0]['min_child_weight']==2000
    counters['default_parameters']+=1
    paths=[Path(runtime.__file__),Path(__file__),ROOT/'sciona/plasticc_weighting.py',ROOT/'docs/reviews/competition_plasticc_source_pins.json',ROOT/'docs/licenses/Avocado-MIT.txt']
    return dict(status='passed',approved=False,checks=counters,lightgbm_version=lgb.__version__,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Actual synthetic grouped fold training; original source and native API comparison',
        adaptations=['Legacy fit verbose/early_stopping arguments replaced with callbacks'],
        limitations=['Synthetic capacity overrides disclosed; source defaults separately checked',
                     'No historical LightGBM binary parity or competition accuracy claim','Full graph remains incomplete'])

if __name__=='__main__':
    result=validate(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/private/tmp/sciona_plasticc_source'))
    (ROOT/'docs/reviews/competition_plasticc_training.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['checks'],sort_keys=True))
