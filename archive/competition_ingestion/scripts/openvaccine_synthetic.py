"""Generated synthetic populations only; no competition records."""
import numpy as np
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS


def payload():
    rng=np.random.default_rng(174)
    def sequences(n,length):return [''.join(rng.choice(list('ACGU'),size=length)) for _ in range(n)]
    labeled=sequences(4,12);pseudo=sequences(2,16);prediction=sequences(2,14)
    targets=rng.normal(size=(4,12,5)).tolist()
    for row in targets:
        for i in range(8,12):row[i]=[None]*5
    splits=[];plans=[]
    for family,slot in sorted(EXPECTED_MEMBERS):
        train,validation=([0,1],[2,3]) if slot%2==0 else ([2,3],[0,1])
        splits.append(dict(family=family,slot=slot,train=train,validation=validation))
        plans.append(dict(family=family,slot=slot,supervised_reverse_flags=[True,False],
            eligible=np.ones((2,16,5),dtype=bool).tolist(),maximum_uncertainty=100.,
            perturbations=np.zeros((2,16,5)).tolist(),sample_weights=[1.,1.],reverse_flags=[False,True]))
    return dict(version=1,seed=582,sequences=labeled,targets=targets,cluster_ids=[0,0,1,2],
        proximity_factors=[1.]*4,splits=splits,pseudo_sequences=pseudo,prediction_sequences=prediction,
        pseudo_rounds=[plans],pretraining_steps=1,initial_supervised_steps=1,
        rollback_policy='model_and_optimizer',absolute_tolerance=0.,std_ddof=0)
