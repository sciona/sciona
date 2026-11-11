"""Qualify source-size boosting controls on synthetic replays only."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from santa_synthetic import payload
from sciona.santa_boosting import fit_threshold_models
from sciona.santa_simulation import simulate


def main():
    p=payload()
    models=fit_threshold_models(p['training'],p['validation'],seed=42,max_rows=10000)
    for model in (models.normal,models.transformed):
        assert model.params['num_leaves']==4095
        assert model.params['learning_rate']==.05
        assert model.params['feature_fraction']==.9 and model.params['bagging_fraction']==.5
        assert model.params['bagging_freq']==5
    for e in models.evaluation.values():
        assert 1<=e['best_iteration']<=1024 and len(e['iteration_rmse'])<=1024
        assert e['best_iteration']==min(range(len(e['iteration_rmse'])),key=e['iteration_rmse'].__getitem__)+1
    game=simulate(models,[50]*100,seed=19)
    assert game['rounds']==2000 and game['total_selections']==4000
    files=['sciona/santa_boosting.py','sciona/santa_simulation.py','scripts/santa_synthetic.py','scripts/validate_santa_default_training.py']
    report=dict(status='passed',approved=False,controls=dict(num_leaves=4095,num_boost_round=1024,stopping_rounds=50),
        rows=models.rows,best_iterations={name:e['best_iteration'] for name,e in models.evaluation.items()},
        evaluated_iterations={name:len(e['iteration_rmse']) for name,e in models.evaluation.items()},
        checks=dict(default_source_controls=True,selected_minimum_heldout_rmse=True,full_game_rounds=2000),
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        scope='Source-size controls on small synthetic populations; no historical data-volume, weights, strength or throughput claim.')
    (ROOT/'docs/reviews/competition_santa2020_default_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['controls','best_iterations','evaluated_iterations','checks']}))


if __name__=='__main__':main()
