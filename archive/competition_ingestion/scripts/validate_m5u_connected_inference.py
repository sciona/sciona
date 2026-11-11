"""Synthetic history-to-trained-quantile-to-repeated-forecast integration check."""
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.m5u_population import build
from sciona.m5u_sampling import sample
from sciona.m5u_targets import assemble
from sciona.m5u_bagging import train
from sciona.m5u_inference import forecast
from sciona.m5u_preparation import prepare


def execute(level):
    rng=np.random.default_rng(853)
    n=1150
    history=4+rng.uniform(0,3,(n,2))+np.sin(np.arange(n)[:,None]/7)
    dates=[date(2000,2,1)+timedelta(days=i) for i in range(n+28)]
    calendar=dict(dates=[d.isoformat() for d in dates],holidays=np.zeros((n+28,1)),
                  state_codes=[0],events=np.zeros((n+28,1)))
    prepared=prepare(history,np.full_like(history,2.),[[0,0,0,0,0],[0,0,0,0,1]],
                     {0:13},controller_level=13 if level==13 else -1,
                     max_level=None if level==13 else 11,
                     normalization='source' if level==13 else 'retained_base_reference')
    history=prepared['history']
    population=build(history,prepared['scaled_history'],prepared['revenue'],
                     prepared['roles'],prepared['levels'],
                     [d.year for d in dates],[d.month for d in dates],299)
    rows,horizons=sample(population['weights'],population['levels'],level,.6,seed=28)
    bag=assemble(population,rows,horizons,history,calendar,-1,scale_range=.7,seed=96)
    quantiles=[.005,.025,.165,.25,.5,.75,.835,.975,.995]
    weights=[.1,.2,.6,.8,1,.9,.7,.2,.1]
    models=train([bag],quantiles,weights,13 if level==13 else -1,single_fold=True,iterations=1,seed=436)
    result=forecast(models,population,history,calendar,level,n-1,quantiles,
                    scale_range=.7,capacity=112,seed=827)
    series_count=int(np.sum(prepared['levels']==level))
    assert result['predictions'].shape==(9,28,series_count)
    assert np.isfinite(result['predictions']).all()
    assert result['batches']==(1 if level<=9 else 28) and result['query_rows']>28*series_count
    assert all(m.report['outer_group_excluded'] for m in models)
    return dict(synthetic_only=True,level=level,normalization=prepared['normalization'],
                            models=len(models),quantiles=9,horizons=28,
                            features=len(bag['features'].columns),candidate_fits=sum(m.report['candidate_fits'] for m in models),
                            query_rows=result['query_rows'],query_batches=result['batches'],finite_predictions=True,
                            outer_group_isolation=True)


def main():
    checks=[execute(level) for level in [1,11,13]]
    paths=sorted(ROOT.glob('sciona/m5u_*.py'))+[ROOT/'sciona/m5u_search_parameters.json',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,checks=checks,
                sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                scope='Connected raw hierarchy preparation through inference at levels 1, 11 and 13, with explicit aggregate normalization correction. One search candidate and reduced inference budget; complete source schedule, graph and publication gates remain pending.')
    (ROOT/'docs/reviews/competition_m5_uncertainty_connected_inference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
