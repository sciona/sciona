"""Execute all 220 M5 family models on synthetic hierarchical time series."""
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.m5_preprocessing import prepare,GROUPINGS
from sciona.m5_pooled_training import train
from sciona.m5_forecast import predict


def configuration():
    rng=np.random.default_rng(749)
    length=250;first=710;count=70
    days=np.arange(first,first+length+28)
    weeks=(days-first)//7
    roles=np.array([[outlet%3,outlet,department%3,department,department]
                    for outlet in range(10) for department in range(7)])
    base=np.arange(count)%9+2
    history=rng.poisson(base[:,None]+np.sin(np.arange(length)[None,:]/9)*.3).astype(float)
    pairs=np.arange(count)
    price_weeks=np.unique(weeks)
    price_pair=np.tile(pairs,len(price_weeks))
    calendar=dict(day=days,week=weeks,
        date=[(date(2000,1,1)+timedelta(days=int(day-first))).isoformat() for day in days],
        categories=[['a' if i%13==0 else None,'b' if i%13==0 else None,None,None,i%2,(i+1)%2,0] for i in range(len(days))])
    prices=dict(pair=price_pair,outlet=roles[price_pair,1],product=roles[price_pair,4],
        week=np.repeat(price_weeks,count),value=(2+price_pair%7).astype(float))
    return dict(history=history,first_day=first,pairs=pairs,roles=roles,
                calendar=calendar,prices=prices,groupings=GROUPINGS)


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    paths=sorted((ROOT/'sciona').glob('m5_*.py'))+[Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    print('Preparing synthetic hierarchy for all six model families',flush=True)
    prepared=prepare(**configuration())
    print('Training 220 models at 3000 configured rounds each',flush=True)
    models=train(prepared,959)
    assert len(models)==220
    counts=[sum((m.recursive,m.pooling)==(recursive,pooling) for m in models)
            for recursive in (True,False) for pooling in ('outlet','outlet_category','outlet_department')]
    assert counts==[10,30,70,10,30,70]
    assert all(m.report['evaluated_rounds']==3000 for m in models)
    print('Executing 28-day recursive and nonrecursive forecasts',flush=True)
    result=predict(prepared,models,959)
    assert result['families'].shape==(6,70,28)
    assert np.isfinite(result['forecast']).all() and (result['forecast']>=0).all()
    np.testing.assert_equal(result['forecast'],sum(result['families'])/6)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(status='passed',approved=False,catalog_mutations=0,
        checks=dict(synthetic_only=True,model_families=6,models=220,family_model_counts=counts,
                    configured_rounds_per_model=3000,evaluated_rounds_total=sum(m.report['evaluated_rounds'] for m in models),
                    horizon=28,synthetic_series=70,finite_forecast=True,exact_mean=True,code_unchanged=True),
        retained_iterations=dict(minimum=min(m.report['retained_iterations'] for m in models),maximum=max(m.report['retained_iterations'] for m in models)),
        sha256=before,
        scope='Complete model-count inventory and full controls on synthetic reduced history/series population. No historical sample-size, accuracy, native parity or production qualification.')
    (ROOT/'docs/reviews/competition_m5_full_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
