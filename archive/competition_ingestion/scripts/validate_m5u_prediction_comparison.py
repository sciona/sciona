"""Compare forecast blending and level adjustment against pinned public software."""
import argparse
import ast
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.m5u_prediction import predict
from sciona.m5u_harmonization import restore
from sciona.m5u_bagging import train


class SyntheticModel:
    feature_name_=['x','z']
    def __init__(self,rng):
        self.coefficients=rng.normal(size=2)
        self.offset=rng.normal()
    def predict(self,frame):
        return frame.to_numpy()@self.coefficients+self.offset


def main(source):
    pins_path=ROOT/'docs/reviews/competition_m5_uncertainty_source_pins.json'
    pins=json.loads(pins_path.read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest==next(v for k,v in pins['files'].items() if k.endswith('quantiles_sage_1to9_eval.py'))
    tree=ast.parse(source.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predictOOS']
    assert len(nodes)==1
    namespace={'np':np,'datetime':datetime,'SINGLE_FOLD':False}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-forecast-function>','exec'),namespace)
    # Execute only the two arithmetic loops, never the source's top-level I/O.
    totals_nodes=[n for n in tree.body if isinstance(n,ast.For) and 2850<n.lineno<2890]
    adjustment_nodes=[n for n in tree.body if isinstance(n,ast.For) and 2890<n.lineno<2910]
    assert len(totals_nodes)==len(adjustment_nodes)==1
    totals_code=compile(ast.Module(body=totals_nodes,type_ignores=[]),'<pinned-level-totals>','exec')
    adjustment_code=compile(ast.Module(body=adjustment_nodes,type_ignores=[]),'<pinned-level-adjustment>','exec')
    rng=np.random.default_rng(492)
    quantiles=[.005,.025,.165,.25,.5,.75,.835,.975,.995]
    comparisons=0
    for case in range(100):
        bag_count=int(rng.integers(1,5));group_count=int(rng.integers(1,5))
        source_models=[[[SyntheticModel(rng) for q in quantiles] for g in range(group_count)] for b in range(bag_count)]
        models=[SimpleNamespace(bag=b,group=g,quantile=q,model=source_models[b][g][i])
                for b in range(bag_count) for g in range(group_count) for i,q in enumerate(quantiles)]
        frame=pd.DataFrame(rng.normal(size=(17,2)),columns=['x','z'])
        scales=rng.uniform(.1,8.,len(frame))
        namespace['SINGLE_FOLD']=group_count==1
        for validation in (False,True):
            with contextlib.redirect_stdout(io.StringIO()):
                expected=namespace['predictOOS'](frame,pd.DataFrame({'scaler':scales}),source_models,quantiles,validation)
            np.testing.assert_allclose(predict(models,frame,scales,quantiles,validation=validation),expected,rtol=1e-12,atol=1e-12)
            comparisons+=1
    # Native fitted models also pass through both implementations, all nine quantiles.
    x=pd.DataFrame(rng.normal(size=(150,2)),columns=['x','z'])
    fitted=train([dict(features=x,targets=3+x.x.to_numpy()*.4,groups=np.repeat([1,2,3],50))],
                 quantiles,[1.]*9,12,iterations=1,seed=341)
    nested=[[[next(m.model for m in fitted if m.group==g and m.quantile==q)
              for q in quantiles] for g in [1,2,3]]]
    namespace['SINGLE_FOLD']=False
    for validation in (False,True):
        with contextlib.redirect_stdout(io.StringIO()):
            expected=namespace['predictOOS'](frame,pd.DataFrame({'scaler':scales}),nested,quantiles,validation)
        np.testing.assert_allclose(predict(fitted,frame,scales,quantiles,validation=validation),expected,rtol=1e-12,atol=1e-12)
        comparisons+=1
    for case in range(100):
        levels=np.repeat(np.arange(1,13),2)
        factors={level:float(rng.uniform(.1,2)) for level in range(1,13)}
        values=rng.uniform(.1,20,size=(9,28,24))
        rows={level:pd.DataFrame(values[:,:,levels==level].reshape(9,-1).T,columns=quantiles)
              .assign(days_fwd=np.repeat(np.arange(1,29),2)) for level in range(1,13)}
        adjustment=.7 if case%2 else 1.
        scope=dict(all_predictions=rows,level_multiplier=factors,LEVEL_QUANTILES={k:quantiles for k in rows},
                   ADJUSTMENT_FACTOR=adjustment,a=pd.DataFrame(index=range(1,29)))
        exec(totals_code,scope);exec(adjustment_code,scope)
        expected=np.concatenate([rows[k][quantiles].to_numpy().T.reshape(9,28,2)/factors[k] for k in range(1,13)],axis=2)
        actual=restore(values,levels,np.array([factors[k] for k in levels]),quantiles,adjustment=adjustment)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    paths=[ROOT/'sciona/m5u_prediction.py',ROOT/'sciona/m5u_harmonization.py',
           ROOT/'tests/test_m5u_prediction.py',ROOT/'tests/test_m5u_harmonization.py',
           ROOT/'sciona/m5u_bagging.py',ROOT/'sciona/m5u_sampling.py',ROOT/'sciona/m5u_search.py',
           ROOT/'sciona/m5u_loss.py',ROOT/'sciona/m5u_search_parameters.json',Path(__file__).resolve(),pins_path]
    report=dict(status='passed',approved=False,catalog_mutations=0,source_software_sha256=digest,
        checks=dict(synthetic_only=True,forecast_comparisons=comparisons,hierarchy_comparisons=100,
                    native_quantiles=9,native_models=len(fitted),
                    candidate_fits=sum(m.report['candidate_fits'] for m in fitted),refits=len(fitted)),
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Forecast blending and final level adjustment only; complete source schedule, inference repeats and graph execution remain unqualified.')
    (ROOT/'docs/reviews/competition_m5_uncertainty_prediction_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    main(parser.parse_args().source)
