"""Integrated source extension parity with real candidate/neighbor implementations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from string import ascii_lowercase
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sciona.trackml_candidates import Candidates
from sciona.trackml_corrections import HelixCorrector
from sciona.trackml_detector import discover_detector
from sciona.trackml_ranking import RankedTrackExtension as TrackExtension
from sciona.trackml_intersections import Intersector
from sciona.trackml_neighbors import Neighbors
from scripts.validate_trackml_detector import synthetic_modules
from scripts.validate_trackml_layer_integration import source_namespace
from scripts.validate_trackml_neighbors import synthetic_hits, LegacyFrame


class ExtensionMaps:
    """Analytic software fixtures; correction grids are 3D, error/pair grids 2D."""
    functions=('dp0','dp1','d_utheta_abs','d_uphi_abs','dnt_utheta_abs','dnt_uphi_abs',
               'pair_theta','pair_phi')

    def __init__(self, missing=False):
        self.missing=missing

    def getGridValues(self, *, is_cylinder, layer_id, functions):
        grids=[]
        bases=dict(dp0=.2,dp1=-.3,d_utheta_abs=100.,d_uphi_abs=120.,
                   dnt_utheta_abs=10.,dnt_uphi_abs=12.,pair_theta=10.,pair_phi=12.)
        for name in functions:
            if self.missing and layer_id==1:
                grids.append((None,None));continue
            ndim=3 if name in ('dp0','dp1') else 2
            axes=[np.array([-1000.,0.,1000.]) for _ in range(ndim)]
            mesh=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
            values=bases[name]+.00001*mesh.sum(axis=-1)+.001*layer_id
            grids.append((axes,values))
        return grids


def validate(root,source):
    ns=source_namespace(root,source);ns['ascii_lowercase']=ascii_lowercase
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    methods={'evaluateLocalBayesianValue','evaluateLocally','evaluateCellFeatureConsistency','evaluateTracks','dropRedundantTracks','filterInvalidTrackCandidates','fitTracks','shortStats','getHelixParams','projectNeighborDisplacement','predictRandomError',
             'bayesianNeighborEvaluation','dropNeighborsInconsistentWithCellFeatures',
             'findHelixIntersectionNeighbors','chooseLikelyNextHits'}
    for filename in ['geometry.py','candidates.py','algorithm.py']:
        path=source/'trackml_solution'/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/'+filename]
        tree=ast.parse(path.read_text())
        if filename=='geometry.py':
            nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
                   {'helixDirectionFromTwoPoints','helixNearestPointDistance','helixUnitTangentVector','circleFromThreePoints','helixPitchLeastSquares'}]
        elif filename=='candidates.py':
            nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        else:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Algorithm')
            nodes=[ast.ClassDef(name='OriginalExtension',bases=[],keywords=[],
                   body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods],decorator_list=[])]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<original-'+filename+'>','exec'),ns)
    modules=synthetic_modules(scale=10)
    with threadpool_limits(limits=1):
        spec=discover_detector(modules)
        ospec=ns['DetectorSpec'](SimpleNamespace(detectors_df=modules))
    counts=dict(search_cases=0,selected_rows=0,paired_hits=0,halfturn_rejections=0,update_cases=0,extended_candidates=0,map_queries=0,bayesian_searches=0,pair_map_searches=0,follow_rounds=0,fit_filter_cases=0,fit_rejections=0,ranking_cases=0,redundancy_cases=0)
    def arrays(a,b):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    def state(a,b):
        arrays([a.open,a.ncross,a._candidates,a._fit],[b.open,b.ncross,b._candidates,b._fit])
        pd.testing.assert_frame_equal(a.df,b.df)
    for seed in range(4):
        rng=np.random.RandomState(1531+seed);n=24
        hits=synthetic_hits(modules,1631+seed)
        xm=rng.uniform(-80,80,n);ym=rng.uniform(-80,80,n);radius=rng.uniform(60,200,n)
        phase=rng.uniform(-np.pi,np.pi,n);pitch=rng.uniform(100,1000,n)*rng.choice([-1.,1.],n)
        zbase=rng.uniform(-100,100,n)
        points=[(xm+radius*np.cos(phase+d),ym+radius*np.sin(phase+d),zbase+pitch*d/(2*np.pi)) for d in [0.,.1,.2]]
        first=len(hits)+1
        seed_ids=np.arange(first,first+3*n).reshape(3,n)
        coords=np.full((first+3*n,3),np.nan);coords[1:first]=hits[['x','y','z']].to_numpy()
        for ids,point in zip(seed_ids,points):coords[ids]=np.array(point).T
        event=SimpleNamespace(hitCoordinatesById=lambda ids:tuple(coords[np.asarray(ids)].T.copy()),
                              hitModuleIdById=lambda ids:np.asarray(ids)%19)
        matrix=np.zeros((n,6),dtype=np.int32);matrix[:,::2]=seed_ids.T
        frame=pd.DataFrame(dict(xf=points[-1][0],yf=points[-1][1],zf=points[-1][2],
            dphi=np.zeros(n),hel_s=np.zeros(n),nskipped=np.zeros(n,dtype=np.int16),dist=np.zeros(n)))
        params={'pair__dist_threshold':10.,'pair__diff_threshold':10.,'pair__cut':2.,
                'follow__weird_triples':False,'nb__dist_threshold':10.,'nb__dist_trust':.5,
                'value__p0':.1,'value__bayes_weight':1.,'value__ploss_weight':.1,'value__ploss_bias':.01,
                'value__fit_weight_hcs':.5,'value__hit_bonus':1.,'value__cross_bonus':1.,'fit__hel_r_min':1.,'follow__nskip_max':2,'nb__cut_factor':1.,'nb__cells_cut':.5}
        for maps in [None,ExtensionMaps(),ExtensionMaps(missing=True)]:
            with threadpool_limits(limits=1):
                an=Neighbors(spec,layer_functions=maps);bn=ns['Neighbors'](ospec,layer_functions=maps)
                hit_in_cyl=np.zeros(len(coords),dtype=bool);hit_in_cyl[first:]=True
                hit_layer_id=np.zeros(len(coords),dtype=int)
                an.fit(hits,hit_in_cyl=hit_in_cyl,hit_layer_id=hit_layer_id);bn.fit(LegacyFrame(hits))
            if maps is not None:
                original_evaluate=an.evaluateLayerFunctions
                def evaluate(*args,**kwargs):
                    counts['map_queries']+=1
                    return original_evaluate(*args,**kwargs)
                an.evaluateLayerFunctions=evaluate
            def setup(original=False):
                cls=ns['Candidates'] if original else Candidates
                candidates=cls(event,nmax_per_crossing=2)
                candidates.initialize(matrix.copy(),frame.copy(deep=True))
                for col,values in [('hel_xm',xm),('hel_ym',ym),('hel_r',radius),('hel_pitch',pitch)]:
                    candidates.setFit(-1,col,values)
                if original:
                    algo=ns['OriginalExtension']();algo.spec=ospec
                    algo.intersector=ns['Intersector'](ospec);algo.log=lambda *args:None
                else:algo=TrackExtension(spec,Intersector(spec))
                run=SimpleNamespace(event=event,candidates=candidates,neighbors=bn if original else an,
                                    params=params.copy(),cell_features=None,layer_functions=maps,
                                    hit_in_cyl=hit_in_cyl,hit_layer_id=hit_layer_id,
                                    hasLayerFunction=lambda name: maps is not None and name in maps.functions)
                return algo,run
            for corrected in [False,True]:
                for revisit in [False,True]:
                    for masked in [False,True]:
                        aa,ar=setup();ba,br=setup(True)
                        mask=None if not masked else np.arange(n)%3!=0
                        ac=HelixCorrector(aa,ar,mask=mask) if corrected else None
                        bc=ns['HelixCorrector'](ba,br,mask=mask) if corrected else None
                        last=ar.candidates.hitCoordinates(-1,mask=mask)
                        lastphase=np.zeros(len(last[0]));lastphase[::5]=np.pi
                        ad=SimpleNamespace();bd=SimpleNamespace()
                        options=dict(last_dphi=lastphase,k=4,nmax_per_crossing=2,revisit=revisit,mask=mask)
                        with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                            actual=aa.findHelixIntersectionNeighbors(ar,-3,-2,-1,*last,corrector=ac,details=ad,**options)
                            expected=ba.findHelixIntersectionNeighbors(br,-3,-2,-1,*last,corrector=bc,details=bd,**options)
                        arrays(actual[:-1],expected[:-1])
                        if actual[-1] is None:assert expected[-1] is None
                        else:
                            pd.testing.assert_frame_equal(actual[-1],expected[-1])
                            counts['selected_rows']+=len(actual[-1])
                            counts['paired_hits']+=int((actual[-1]['extend_hit_id_b']!=0).sum())
                        assert vars(ad).keys()==vars(bd).keys()
                        for key in vars(ad):np.testing.assert_array_equal(getattr(ad,key),getattr(bd,key))
                        if corrected:arrays(ac.helixParams(),bc.helixParams())
                        counts['halfturn_rejections']+=int(ad.nrejected_dphi)
                        counts['search_cases']+=1
                        if maps is not None:
                            if revisit: counts['pair_map_searches']+=1
                            elif hasattr(ad,'bay_cut'): counts['bayesian_searches']+=1
            for skips in [0,3]:
                aa,ar=setup();ba,br=setup(True)
                ar.candidates.df['nskipped']=np.full(n,skips,dtype=np.int16)
                br.candidates.df['nskipped']=np.full(n,skips,dtype=np.int16)
                with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                    aa.chooseLikelyNextHits(ar,k=4,nmax_per_crossing=2)
                    ba.chooseLikelyNextHits(br,k=4,nmax_per_crossing=2)
                state(ar.candidates,br.candidates)
                counts['update_cases']+=1
                extended=ar.candidates.ncross>3
                counts['extended_candidates']+=int(extended.sum())
                if skips==3:assert not extended.any()
                else:
                    assert (ar.candidates.df.loc[extended,'nskipped']==0).all()
                    assert (ar.candidates.df.loc[extended,'dphi']==0).all()
            # Refit each newly extended population before the next following step.
            aa,ar=setup();ba,br=setup(True)
            for round_index in range(3):
                assert ar.candidates.n==br.candidates.n
                if ar.candidates.n==0:break
                with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                    aa.fitTracks(ar,step=round_index);ba.fitTracks(br,step=round_index)
                    state(ar.candidates,br.candidates)
                    if ar.candidates.n==0:break
                    aa.chooseLikelyNextHits(ar,k=4,nmax_per_crossing=2,step=round_index)
                    ba.chooseLikelyNextHits(br,k=4,nmax_per_crossing=2,step=round_index)
                state(ar.candidates,br.candidates)
                counts['follow_rounds']+=1
                if ar.candidates.n:
                    aa.fitTracks(ar);ba.fitTracks(br)
                    av,af=aa.evaluateTracks(ar,analysis=True)
                    bv,bf=ba.evaluateTracks(br,analysis=True)
                    np.testing.assert_array_equal(av,bv);pd.testing.assert_frame_equal(af,bf)
                    assert np.isfinite(av).all()
                    order=np.argsort(-av)
                    ar.candidates.permute(order);br.candidates.permute(np.argsort(-bv))
                    af=af.iloc[order].reset_index(drop=True);bf=bf.iloc[order].reset_index(drop=True)
                    before_hits=ar.candidates._candidates.copy()
                    seen=set();keep=[]
                    for row in before_hits:
                        key=tuple(sorted(row[:4]));keep.append(key not in seen);seen.add(key)
                    af=aa.dropRedundantTracks(ar,af,nlayers=2)
                    bf=ba.dropRedundantTracks(br,bf,nlayers=2)
                    state(ar.candidates,br.candidates);pd.testing.assert_frame_equal(af,bf)
                    np.testing.assert_array_equal(ar.candidates._candidates,before_hits[np.asarray(keep)])
                    counts['ranking_cases']+=1;counts['redundancy_cases']+=1
            for masked_fit in [False,True]:
                aa,ar=setup();ba,br=setup(True)
                ar.params['fit__hel_r_min']=130.;br.params['fit__hel_r_min']=130.
                fitmask=None if not masked_fit else np.arange(n)%2==0
                aa.fitTracks(ar,mask=fitmask);ba.fitTracks(br,mask=fitmask)
                state(ar.candidates,br.candidates)
                keep_expected=radius>=130.
                if fitmask is not None:keep_expected|=~fitmask
                np.testing.assert_array_equal(ar.candidates._candidates,matrix[keep_expected])
                counts['fit_filter_cases']+=1
                counts['fit_rejections']+=n-ar.candidates.n
    assert counts['extended_candidates']>0 and counts['paired_hits']>0 and counts['halfturn_rejections']>0
    assert counts['map_queries']>0 and counts['bayesian_searches']>0 and counts['pair_map_searches']>0
    paths=['sciona/trackml_ranking.py','sciona/trackml_seeding.py','sciona/trackml_fitting.py','sciona/trackml_extension.py','sciona/trackml_candidates.py','sciona/trackml_neighbors.py',
           'sciona/trackml_intersections.py','sciona/trackml_corrections.py','sciona/trackml_bayesian.py',
           'sciona/trackml_nearest.py','sciona/trackml_helix.py','sciona/trackml_geometry.py',
           'sciona/trackml_detector.py','sciona/trackml_layer_functions.py',
           'scripts/validate_trackml_extension.py','scripts/validate_trackml_neighbors.py',
           'scripts/validate_trackml_layer_integration.py','scripts/validate_trackml_detector.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Fallback and Bayesian/pair-map branches exercised using analytic synthetic grids, not learned maps.',
                     'Correction/refitting integrated with nonzero and missing-layer grids; no learned accuracy claim.',
                     'Repeated fit/follow rounds validated; cell-enabled extension, seed discovery and complete tracking lifecycle remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_extension.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
