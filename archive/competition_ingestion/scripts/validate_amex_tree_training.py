"""Execute both downstream DART variants at the recovered full round count."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import amex_tree as tree


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    files=[ROOT/'sciona/amex_tree.py',ROOT/'sciona/amex_numeric.py',ROOT/'tests/test_amex_tree.py',Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={str(p.relative_to(ROOT)):sha(p) for p in files}
    rng=np.random.default_rng(111);labels=[0]*800+[1]*800
    x=rng.normal(size=(1600,40))+np.asarray(labels)[:,None];q=rng.normal(size=(8,40))
    slots=rng.uniform(size=(1600,13));query_slots=rng.uniform(size=(8,13));slots[:20,:3]=np.nan
    native=tree.lgb.train;counts=[]
    def record(params,dataset,**kwargs):
        assert kwargs['num_boost_round']==4500 and params['boosting']=='dart' and params['feature_fraction']==.05
        history={};kwargs['callbacks']=[tree.lgb.record_evaluation(history)]
        model=native(params,dataset,**kwargs)
        count=len(history['valid_1']['binary_logloss']);assert count==4500
        counts.append(count);print('Completed downstream DART fit '+str(len(counts)),flush=True)
        return model
    with patch.object(tree.lgb,'train',side_effect=record):result=tree.fit(x,slots,labels,q,query_slots)
    assert result['models']==10 and counts==[4500]*10
    for variant in result['variants'].values():
        assert np.isfinite(variant['training_predictions']).all()
        assert np.all((variant['query_predictions']>=0)&(variant['query_predictions']<=1))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in files}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,models=10,variants=2,configured_rounds=4500,evaluated_rounds=counts,
        native_iterations={k:v['iterations'] for k,v in result['variants'].items()},sha256=before,
        limits=['Synthetic manual features and probability slots; upstream whole-pipeline integration remains pending.',
            'Independent current CPU runtime, forty manual features and synthetic population; no historical volume or competitive accuracy qualification.'])
    (ROOT/'docs/reviews/competition_amex_tree_training.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='passed',models=10,evaluated_rounds=counts)))


if __name__=='__main__':main()
