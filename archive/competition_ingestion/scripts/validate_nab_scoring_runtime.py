"""Compare corrected scoring to pinned NAB across synthetic profiles and windows."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from sciona.nab_scoring import evaluate
ROOT=Path(__file__).resolve().parents[1]


def main():
    pins=json.loads((ROOT/'docs/reviews/competition_nab_source_pins.json').read_text())
    path=Path('/private/tmp/sciona_nab_source/nab/sweeper.py')
    digest=next(p['sha256'] for p in pins['pins'] if p['software_path']=='nab/sweeper.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    spec=importlib.util.spec_from_file_location('pinned_nab_sweeper',path);source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    rng=np.random.default_rng(912);cases=0;maximum_error=0.;boundary_cases=0
    for size in (16,32,61):
        for probation in (0.,.15,.4):
            for costs in [dict(tpWeight=1.,fpWeight=.11,fnWeight=1.),dict(tpWeight=1.,fpWeight=.22,fnWeight=1.),dict(tpWeight=1.,fpWeight=.11,fnWeight=2.)]:
                for windows in [[],[(2,5)],[(1,3),(8,12)],[(size-3,size-1)]]:
                    values=rng.random(size);values[0]=0.;values[-1]=1.
                    for threshold in (0.,.2,.5,1.):
                        sweeper=source.Sweeper(probationPercent=probation,costMatrix=costs)
                        _,expected=sweeper.scoreDataSet(range(size),values,windows,'synthetic',threshold)
                        actual=evaluate(values,windows,threshold=threshold,probation_percent=probation,costs=costs)
                        if expected is None:
                            points=sweeper.calcSweepScore(range(size),values,windows,'synthetic')
                            scorable=source.prepAnomalyListForScoring(points)
                            assert scorable and threshold<min(p.anomalyScore for p in scorable)
                            expected=sweeper.calcScoreByThreshold(points)[-1]
                            boundary_cases+=1
                        error=abs(expected.score-actual['raw_score']);maximum_error=max(maximum_error,error)
                        assert error<1e-12
                        assert actual['counts']=={k:getattr(expected,k) for k in ('tp','tn','fp','fn')}
                        assert actual['scored_points']==expected.total
                        cases+=1
    paths=['sciona/nab_scoring.py','scripts/validate_nab_scoring_runtime.py','tests/test_nab_runtime.py']
    report=dict(status='passed',synthetic_only=True,source_commit=pins['commit'],source_sha256=digest,
        source_comparison_cases=cases,corrected_threshold_selection_cases=boundary_cases,max_absolute_score_error=maximum_error,point_confusion_counts_exact=True,
        checks=['Inclusive windows, tied endpoint thresholds, multiple windows and probation.',
                'Three explicit cost profiles; below-minimum thresholds select the final source sweep row explicitly. Singleton correction is tested separately.'],
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        scope='Corrected scoring component; full streaming graph and publication pending.')
    (ROOT/'docs/reviews/competition_nab_scoring_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',source_comparison_cases=cases,max_absolute_score_error=maximum_error)))


if __name__=='__main__':main()
