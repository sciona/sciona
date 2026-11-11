"""Serialized synthetic runtime checks with distinct prediction objects."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import plasticc_runtime as runtime
from sciona.plasticc_features import plasticc_start_time,plasticc_bands


def synthetic_payload():
    def obj(label,i,prediction=False):
        times=np.linspace(plasticc_start_time,plasticc_start_time+1000,72)
        flux=1000*np.sin((times-plasticc_start_time)/(40+i*10))
        metadata=dict(object_id=('prediction' if prediction else 'training')+'-synthetic-%d-%d'%(label,i),
            host_specz=0 if label==6 else .2,redshift=0 if label==6 else .2,
            host_photoz=0 if label==6 else .22,host_photoz_error=.02,
            galactic=label==6,ddf=False,mwebv=.01,ra=0.,decl=0.)
        if not prediction: metadata['class']=label
        return dict(metadata=metadata,observations=[[float(t),plasticc_bands[j%6],float(f),1.] for j,(t,f) in enumerate(zip(times,flux))])
    return dict(version=1,training=[obj(c,i) for c in [6,42,52,62,95] for i in range(3)],
        prediction=[obj(c,4,True) for c in [6,42]],photoz_reference=[[.1,.12,.02],[.4,.35,.04]],
        config=dict(num_augments=1,num_folds=3,training_parameters=dict(n_estimators=60,n_jobs=1,
            min_child_weight=.01,min_child_samples=2,min_split_gain=0.,random_seed=4)))


def validate():
    payload=json.loads(json.dumps(synthetic_payload(),allow_nan=False))
    result=runtime.predict_runtime(runtime.train_runtime(runtime.prepare_runtime(payload)))
    assert result['object_ids']==[o['metadata']['object_id'] for o in payload['prediction']]
    assert len(result['probabilities'])==2
    np.testing.assert_allclose(np.sum(result['probabilities'],axis=1),1)
    json.dumps(result,allow_nan=False)
    # Second full run exercises redshift and competition weighting, with fresh inputs.
    weighted=copy.deepcopy(payload); weighted['config'].update(weighting='redshift',class_weighting='kaggle')
    result2=runtime.predict_runtime(runtime.train_runtime(runtime.prepare_runtime(weighted)))
    assert result2['object_ids']==result['object_ids']
    json.dumps(result2,allow_nan=False)
    bad=[]
    p=copy.deepcopy(payload); p['version']=True; bad.append(p)
    p=copy.deepcopy(payload); p['config']['num_folds']=4; bad.append(p)
    p=copy.deepcopy(payload); p['config']['training_parameters']['model_file']='invalid'; bad.append(p)
    p=copy.deepcopy(payload); p['prediction'][0]['metadata']['object_id']=p['training'][0]['metadata']['object_id']; bad.append(p)
    p=copy.deepcopy(payload); p['training'][0]['observations'][0][3]=0; bad.append(p)
    p=copy.deepcopy(payload); p['training'][0]['metadata']['galactic']=1; bad.append(p)
    p=copy.deepcopy(payload); p['training'][0]['observations'][0][2]=float('nan'); bad.append(p)
    p=copy.deepcopy(payload); p['photoz_reference']=[]; bad.append(p)
    for p in bad:
        try: runtime.prepare_runtime(p)
        except ValueError: pass
        else: raise AssertionError('Malformed runtime payload accepted')
    paths=[ROOT/'sciona'/('plasticc_'+n+'.py') for n in ['runtime','population','defaults','observations','augmentation','features','weighting','training','predictions']]+[Path(__file__)]
    return dict(status='passed',approved=False,checks=dict(serialized_complete_runs=2,distinct_prediction_objects=2,
        weighting_modes=2,rejected_payloads=len(bad)),hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Actual synthetic serialized training-to-distinct-prediction execution',remaining=['Graph/provider binding and automated promotion review'])

if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/competition_plasticc_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks'],sort_keys=True))
