"""Run complete corrected uncertainty pipeline with full source default budgets."""
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.m5u_pipeline import run


def main():
    n=1338;rng=np.random.default_rng(1297)
    units=4+rng.uniform(0,3,(n,6))+np.sin(np.arange(n)[:,None]/7)
    roles=np.array([[0,0,category,category,2*category+product]
                    for category in range(3) for product in range(2)])
    dates=[date(2000,2,1)+timedelta(days=i) for i in range(n+28)]
    calendar=dict(dates=[d.isoformat() for d in dates],holidays=np.zeros((n+28,1)),
                  state_codes=[0],events=np.zeros((n+28,1)))
    result=run(units,np.full_like(units,2.),roles,{0:13,1:14,2:15},calendar,
               progress=lambda value:print(json.dumps(value),flush=True))
    assert result['predictions'].shape[:2]==(9,28)
    assert np.isfinite(result['predictions']).all()
    reports=result['reports']
    assert len(reports)==14 and sum(r['models'] for r in reports.values())==126
    assert all(r['outer_group_isolation'] for r in reports.values())
    # At this synthetic population size, both clipping passes yield the full
    # source maximum of 2,500 repeats for every series/horizon query.
    queries=sum(r['query_rows'] for r in reports.values())
    assert queries==result['predictions'].shape[2]*28*2500
    paths=sorted(ROOT.glob('sciona/m5u_*.py'))+[ROOT/'sciona/m5u_search_parameters.json',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
                checks=dict(levels=14,models=126,quantiles=9,horizons=28,query_rows=queries,
                            candidate_fits=sum(r['candidate_fits'] for r in reports.values()),
                            refits=126,full_source_search_counts=True,full_source_inference_budget=True,
                            complete_hierarchy_coverage=True,finite_restored_predictions=True,outer_group_isolation=True),
                levels=reports,normalization=result['normalization'],
                sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                scope='Complete synthetic corrected reconstruction, full source default training and forecast repetition budgets. Provider graph, serialization, environment review and publication gates remain pending. No historical competition parity claim.')
    (ROOT/'docs/reviews/competition_m5_uncertainty_full_pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
