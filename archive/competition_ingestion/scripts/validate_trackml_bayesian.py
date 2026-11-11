"""Pinned-source and analytic Bayesian neighbor scoring on synthetic runtime maps."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

from sciona.trackml_bayesian import BayesianNeighborScorer
from sciona.trackml_layer_functions import build_layer_evaluator
from scripts.validate_trackml_layer_integration import SyntheticMaps, source_namespace, source_evaluator


class ErrorMaps(SyntheticMaps):
    functions=('d_utheta_abs','d_uphi_abs','dnt_utheta_abs','dnt_uphi_abs')

    def formula(self,points,cylinder,layer,function):
        bases={'d_utheta_abs':8.,'d_uphi_abs':12.,'dnt_utheta_abs':.1,'dnt_uphi_abs':.2}
        return np.full(len(points),bases[function])+.00001*points.sum(axis=1)+.001*layer


def validate(root,source):
    ns=source_namespace(root,source)
    path=source/'trackml_solution/algorithm.py'
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/algorithm.py']
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Algorithm')
    names={'projectNeighborDisplacement','predictRandomError','bayesianNeighborEvaluation'}
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(methods)==3
    oracle=ast.ClassDef(name='OriginalScorer',bases=[],keywords=[],body=methods,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[oracle],type_ignores=[])),'<original-scorer>','exec'),ns)
    a=BayesianNeighborScorer();b=ns['OriginalScorer']()
    counts=dict(scoring_cases=0,analytic_cases=0,good=0,dubious=0,missing_map_rows=0)
    for seed in range(8):
        rng=np.random.RandomState(1331+seed);n=96
        phi=rng.uniform(-np.pi,np.pi,n);rad=rng.uniform(10.,200.,n)
        xyz=np.array([rad*np.cos(phi),rad*np.sin(phi),rng.uniform(-300.,300.,n)])
        xyz[2,0]=0.  # Source theta projection degenerates to zero in this plane.
        nearest=xyz+rng.normal(size=xyz.shape)*rng.choice([.01,1.,10.],n)
        frame=pd.DataFrame(dict(extend_hit_id=np.arange(n),extend_index=np.arange(n)[::-1],
            xn=nearest[0],yn=nearest[1],zn=nearest[2],cyl_closer=rng.choice([False,True],n),
            next_id=np.arange(n)%3,hel_s=rng.uniform(-100.,100.,n)))
        before=frame.copy(deep=True)
        for missing in [False,True]:
            maps=ErrorMaps(dimensions=2,missing=missing)
            ae=build_layer_evaluator(maps,cylinder_count=3,cap_count=4)
            be=source_evaluator(ns,maps)
            for factor in [.5,1.,2.]:
                for trust in [.5,1.]:
                    params={'nb__cut_factor':factor,'nb__dist_trust':trust}
                    event=SimpleNamespace(hitCoordinatesById=lambda ids: tuple(x[ids] for x in xyz))
                    ar=SimpleNamespace(event=event,neighbors=ae,params=params)
                    br=SimpleNamespace(event=event,neighbors=be,params=params)
                    ad=SimpleNamespace();bd=SimpleNamespace()
                    actual=a.bayesianNeighborEvaluation(ar,frame,details=ad)
                    expected=b.bayesianNeighborEvaluation(br,frame,details=bd)
                    for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
                    assert vars(ad).keys()==vars(bd).keys()
                    for name in vars(ad):np.testing.assert_array_equal(getattr(ad,name),getattr(bd,name))
                    # Independent spherical basis, formal variances and threshold.
                    r=np.linalg.norm(xyz,axis=0)
                    theta=np.arctan2(rad,np.abs(xyz[2]))
                    ut=np.array([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),
                                 -np.sign(xyz[2])*np.sin(theta)])
                    ut[:,xyz[2]==0]=0.
                    up=np.array([-np.sin(phi),np.cos(phi),np.zeros(n)])
                    delta=xyz-nearest
                    dt=np.sum(ut*delta,axis=0);dp=np.sum(up*delta,axis=0)
                    np.testing.assert_allclose(ad.bay_d_theta,dt,rtol=1e-12,atol=1e-13)
                    np.testing.assert_allclose(ad.bay_d_phi,dp,rtol=1e-12,atol=1e-13)
                    fields=ae.evaluateLayerFunctions(*xyz,frame.cyl_closer.to_numpy(),frame.next_id.to_numpy())
                    et=np.hypot(fields[2],.007884*.2096*frame.hel_s.to_numpy())
                    ep=np.hypot(fields[3],.007884*.3853*frame.hel_s.to_numpy())
                    de=np.hypot(dt/et,dp/ep)
                    cut=factor*np.sqrt(2*(np.log1p(2*(fields[0]/et)**2/np.pi)+
                                          np.log1p(2*(fields[1]/ep)**2/np.pi)))
                    np.testing.assert_allclose(ad.bay_e_theta,et,rtol=1e-13,equal_nan=True)
                    np.testing.assert_allclose(ad.bay_e_phi,ep,rtol=1e-13,equal_nan=True)
                    np.testing.assert_allclose(ad.bay_cut,cut,rtol=1e-13,equal_nan=True)
                    np.testing.assert_array_equal(actual[0],de<cut)
                    np.testing.assert_array_equal(actual[1],de>trust*cut)
                    counts['scoring_cases']+=1;counts['analytic_cases']+=1
                    counts['good']+=int(np.sum(actual[0]));counts['dubious']+=int(np.sum(actual[1]))
                    if missing:
                        sel=frame.next_id.to_numpy()==1
                        assert not np.any(np.asarray(actual[0])[sel])
                        assert not np.any(np.asarray(actual[1])[sel])
                        counts['missing_map_rows']+=int(sel.sum())
            pd.testing.assert_frame_equal(frame,before)
    # Strict equality at zero cut: neither good nor dubious for zero displacement.
    frame=frame.copy();frame[['xn','yn','zn']]=xyz.T
    ar.params={'nb__cut_factor':0.,'nb__dist_trust':1.}
    result=a.bayesianNeighborEvaluation(ar,frame)
    assert not np.any(result[0]) and not np.any(result[1])
    assert counts['good']>0 and counts['dubious']>0
    paths=['sciona/trackml_bayesian.py','sciona/trackml_layer_functions.py',
           'scripts/validate_trackml_bayesian.py','scripts/validate_trackml_layer_integration.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Synthetic error/background grids, no learned uncertainty calibration or accuracy claim.',
                     'Source z=0 theta projection is zero; axis singularities remain source arithmetic.',
                     'Missing-map NaNs produce both flags false, not an automatic dubious flag.',
                     'Intersection candidate filtering/extension and track commitment remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_bayesian.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
