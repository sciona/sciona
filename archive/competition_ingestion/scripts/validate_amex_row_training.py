"""Execute all five grouped DART folds with the recovered 4500-round control."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import amex_row_model as row


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    files=[ROOT/'sciona/amex_row_model.py',ROOT/'sciona/amex_numeric.py',ROOT/'tests/test_amex_row_model.py',Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={str(p.relative_to(ROOT)):sha(p) for p in files}
    rng=np.random.default_rng(109);labels=[0]*50+[1]*50
    xs=[rng.normal(loc=2*y,size=(13,4)) for y in labels];qs=[rng.normal(size=(n,4)) for n in (1,7,13)]
    real=row.lgb.train;counts=[]
    def record(params,reference,**kw):
        assert params['boosting']=='dart' and kw['num_boost_round']==4500
        history={};kw['callbacks']=[row.lgb.record_evaluation(history)]
        model=real(params,reference,**kw)
        evaluated=len(history['valid_1']['binary_logloss']);assert evaluated==4500
        counts.append(evaluated);print('Completed native DART fold '+str(len(counts)),flush=True)
        return model
    with patch.object(row.lgb,'train',side_effect=record):result=row.fit(xs,labels,qs)
    assert result['models']==5 and counts==[4500]*5
    assert all(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) for p in result['training_predictions']+result['query_predictions'])
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in files}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,models=5,configured_rounds=4500,evaluated_rounds=counts,native_iterations=result['iterations'],
        source_component='Grouped row DART with recovered hyperparameters and default seed42',sha256=before,
        limits=['Synthetic four-feature row population; historical row volume, precision and winning performance unqualified.',
            'Current deterministic single-thread LightGBM and in-memory booster reload independently replace historical runtime and file transport.',
            'Full downstream neural/tree ensemble and served graph remain pending.'])
    (ROOT/'docs/reviews/competition_amex_row_training.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='passed',models=5,evaluated_rounds=counts)))


if __name__=='__main__':main()
