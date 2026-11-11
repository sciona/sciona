"""Execute recovered training limits while inspecting actual native fold routing."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import vsb_training as training
from sciona.vsb_threshold import repeated_folds,threshold


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    files=sorted((ROOT/'sciona').glob('vsb_*.py'))+[Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(ROOT)):sha(p) for p in files}
    rng=np.random.default_rng(100);x=rng.normal(size=(200,9));q=rng.normal(size=(8,9));labels=np.zeros((200,3),int)
    labels[x[:,0]>0,0]=1;labels[x[:,0]>.5,1:]=1;x[0,2]=np.nan
    y=labels.any(axis=1).astype(int);plans=repeated_folds(y,seed=123948,repetitions=25)
    real=training.lgb.train;observed=[];query_predictions=[]
    def inspect_fit(params,reference,**kwargs):
        index=len(observed);tr,val,te=plans[index];order=[2,7,3,4,8,0,6,5,1]
        assert kwargs['num_boost_round']==10000 and kwargs['valid_names']==['train','test','validation']
        assert params['learning_rate']==.01 and params['num_leaves']==80 and params['seed']==23974
        for dataset,rows in zip(kwargs['valid_sets'],(tr,te,val)):
            np.testing.assert_array_equal(dataset.data,x[rows][:,order]);np.testing.assert_array_equal(dataset.label,y[rows])
        callback=kwargs['callbacks'][0]
        assert callback.stopping_rounds==100 and callback.first_metric_only is False
        events=[]
        def capture(env):events.append((env.iteration,tuple((r[0],r[1]) for r in env.evaluation_result_list)))
        capture.order=15
        kwargs['callbacks']=[*kwargs['callbacks'],capture]
        model=real(params,reference,**kwargs)
        assert events and all(e[1]==(('train','binary_logloss'),('test','binary_logloss'),('validation','binary_logloss')) for e in events)
        observed.append(dict(best=model.best_iteration,evaluated=len(events)))
        query_predictions.append(model.predict(q[:,order]))
        if len(observed)%25==0:print('Completed native fits: '+str(len(observed)),flush=True)
        return model
    with patch.object(training.lgb,'train',side_effect=inspect_fit):result=training.fit(x,labels,q)
    assert result['models']==len(observed)==125
    np.testing.assert_allclose(result['probabilities'],np.mean(query_predictions,axis=0),atol=1e-14,rtol=1e-13)
    assert result['threshold']==threshold(labels.reshape(-1),np.repeat(result['training_probabilities'],3))['threshold']
    np.testing.assert_array_equal(result['signal_decisions'],np.repeat((result['probabilities']>result['threshold'])[:,None],3,axis=1))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in files}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,models=125,round_ceiling=10000,stopping_patience=100,
        evaluated_rounds_min=min(r['evaluated'] for r in observed),evaluated_rounds_max=max(r['evaluated'] for r in observed),
        best_iterations_min=min(r['best'] for r in observed),best_iterations_max=max(r['best'] for r in observed),
        checks=dict(all_fold_features_and_labels_verified=True,both_holdouts_in_early_stopping=True,query_mean_oracle=True,signal_threshold_oracle=True,code_unchanged=True),sha256=before,
        limits=['Full recovered training controls on synthetic aggregate features; early stopping determines actual rounds.',
            'Independent current CPU LightGBM; no historical native parity, accuracy or untouched holdout qualification.',
            'Complete raw-signal serialized graph remains pending.'])
    (ROOT/'docs/reviews/competition_vsb_full_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sha256','limits')}),flush=True)


if __name__=='__main__':main()
