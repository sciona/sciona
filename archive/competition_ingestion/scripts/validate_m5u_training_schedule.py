"""Execute every configured source training level with full search effort."""
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.m5u_preparation import prepare
from sciona.m5u_population import build
from sciona.m5u_training import train_batch,PARAMETERS,QUANTILES


def main():
    n=1338;rng=np.random.default_rng(1297)
    units=4+rng.uniform(0,3,(n,6))+np.sin(np.arange(n)[:,None]/7)
    roles=np.array([[0,0,category,category,2*category+product]
                    for category in range(3) for product in range(2)])
    dates=[date(2000,2,1)+timedelta(days=i) for i in range(n+28)]
    calendar=dict(dates=[d.isoformat() for d in dates],holidays=np.zeros((n+28,1)),
                  state_codes=[0],events=np.zeros((n+28,1)))
    reports={}
    for controller in [-1,13,14,15]:
        prepared=prepare(units,np.full_like(units,2.),roles,{0:13,1:14,2:15},
                         controller_level=controller,max_level=11 if controller==-1 else None,
                         normalization='retained_base_reference' if controller==-1 else 'source')
        population=build(prepared['history'],prepared['scaled_history'],prepared['revenue'],
                         prepared['roles'],prepared['levels'],[d.year for d in dates],
                         [d.month for d in dates],299)
        models,checked=train_batch(prepared,population,calendar,controller,seed=514+controller)
        for level,items in models.items():
            assert len(items)==9 and {m.quantile for m in items}==set(QUANTILES)
            assert all(m.report['candidates']==checked[level]['iterations'] for m in items)
            assert all(m.report['outer_group_excluded'] for m in items)
        reports.update(checked)
        print(json.dumps(dict(controller=controller,levels=len(checked),models=sum(x['models'] for x in checked.values()),
                              candidate_fits=sum(x['candidate_fits'] for x in checked.values()))),flush=True)
    assert set(reports)==set(PARAMETERS)
    assert sum(x['models'] for x in reports.values())==126
    paths=sorted(ROOT.glob('sciona/m5u_*.py'))+[ROOT/'sciona/m5u_search_parameters.json',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
                checks=dict(levels=len(reports),models=126,quantiles=9,
                            candidate_fits=sum(x['candidate_fits'] for x in reports.values()),
                            refits=sum(x['refits'] for x in reports.values()),full_source_search_counts=True,
                            full_source_sampling_rates=True,outer_group_isolation=True),levels=reports,
                sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                scope='All 14 configured training levels with full default search and sampling schedule on synthetic histories. Explicit aggregate normalization correction applies. Full-budget forecast assembly, provider graph and publication gates remain pending.')
    (ROOT/'docs/reviews/competition_m5_uncertainty_training_schedule.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
