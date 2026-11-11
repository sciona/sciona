"""Source seed discovery through two-hit fitting on synthetic detector hits."""
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
from sciona.trackml_detector import discover_detector
from sciona.trackml_intersections import Intersector
from sciona.trackml_neighbors import Neighbors
from sciona.trackml_pairing import PairedTrackExtension as SeededTrackExtension
from scripts.validate_trackml_detector import synthetic_modules
from scripts.validate_trackml_layer_integration import source_namespace
from scripts.validate_trackml_neighbors import synthetic_hits,LegacyFrame


def validate(root,source):
    ns=source_namespace(root,source);ns['ascii_lowercase']=ascii_lowercase
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    methods={'findPairs','filterInvalidTrackCandidates','chooseLikelyFirstHits','chooseLikelySecondHits','fitTracks','getHelixParams',
             'shortStats','projectNeighborDisplacement','predictRandomError','bayesianNeighborEvaluation',
             'findHelixIntersectionNeighbors','chooseLikelyNextHits','dropNeighborsInconsistentWithCellFeatures'}
    for filename in ['geometry.py','candidates.py','algorithm.py']:
        path=source/'trackml_solution'/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/'+filename]
        tree=ast.parse(path.read_text())
        if filename=='geometry.py':nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        elif filename=='candidates.py':nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        else:
            c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Algorithm')
            nodes=[ast.ClassDef(name='Original',bases=[],keywords=[],decorator_list=[],
                   body=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in methods])]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<original-'+filename+'>','exec'),ns)
    counts=dict(seed_cases=0,seed_pairs=0,fit_cases=0,follow_cases=0,three_hit_candidates=0,pair_rounds=0,pair_survivors=0,filled_pair_slots=0,double_pair_rounds=0)
    modules=synthetic_modules(scale=10)
    with threadpool_limits(limits=1):
        spec=discover_detector(modules);ospec=ns['DetectorSpec'](SimpleNamespace(detectors_df=modules))
    def state(a,b):
        for name in ['open','ncross','_candidates','_fit']:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        pd.testing.assert_frame_equal(a.df,b.df)
    for seed in range(4):
        hits=synthetic_hits(modules,1831+seed)
        # Place synthetic barrel hits near the midplane to permit three crossings.
        hits.loc[hits.volume_id==8,'z']*=.1
        xyz=np.full((len(hits)+1,3),np.nan);xyz[1:]=hits[['x','y','z']].to_numpy()
        event=SimpleNamespace(hitCoordinatesById=lambda ids:tuple(xyz[np.asarray(ids)].T.copy()),
                              hitModuleIdById=lambda ids:np.asarray(ids)%19)
        with threadpool_limits(limits=1):
            an=Neighbors(spec);bn=ns['Neighbors'](ospec)
            hit_in_cyl=np.zeros(len(xyz),dtype=bool);hit_layer_id=np.zeros(len(xyz),dtype=int)
            an.fit(hits,hit_in_cyl=hit_in_cyl,hit_layer_id=hit_layer_id);bn.fit(LegacyFrame(hits))
        for nlayers in [1,2]:
            for origin in [(0.,0.,0.),(1.,-2.,3.)]:
                params={'nb__origin_dz':10.,'nb__nlayers':nlayers,'nb__radius_exp':1.,
                        'nb__cyl_origin_area':100.,'nb__cyl_scale':5.,'nb__cap_origin_radius':10.,
                        'fit__hel_r_min':1.,'pair__dist_threshold':10.,'pair__diff_threshold':10.,
                        'follow__weird_triples':False,'nb__dist_threshold':10.,'nb__dist_trust':.5,
                        'follow__nskip_max':2}
                aa=SeededTrackExtension(spec,Intersector(spec));ba=ns['Original']()
                ba.spec=ospec;ba.intersector=ns['Intersector'](ospec);ba.log=lambda *args:None
                ar=SimpleNamespace(event=event,neighbors=an,params=params,cell_features=None,layer_functions=None,
                                   hit_in_cyl=hit_in_cyl,hit_layer_id=hit_layer_id,
                                   hasLayerFunction=lambda name:False,candidates=Candidates(event,nmax_per_crossing=2))
                br=SimpleNamespace(event=event,neighbors=bn,params=params,cell_features=None,layer_functions=None,
                                   hit_in_cyl=hit_in_cyl,hit_layer_id=hit_layer_id,
                                   hasLayerFunction=lambda name:False,candidates=ns['Candidates'](event,nmax_per_crossing=2))
                first=aa.chooseLikelyFirstHits(ar);ofirst=ba.chooseLikelyFirstHits(br)
                pd.testing.assert_frame_equal(pd.DataFrame(first),pd.DataFrame(ofirst))
                # Small deterministic query population; second-hit results are not truncated.
                ids=first.hit_id.iloc[:12]
                with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                    pairs=aa.chooseLikelySecondHits(ar,ids,origin_coords=origin)
                    opairs=ba.chooseLikelySecondHits(br,ids,origin_coords=origin)
                assert pairs is not None
                pd.testing.assert_frame_equal(pairs,opairs)
                assert pairs.hit_id.isin(ids).all() and pairs.nb_hit_id.isin(hits.hit_id).all()
                counts['seed_cases']+=1;counts['seed_pairs']+=len(pairs)
                last=event.hitCoordinatesById(pairs.nb_hit_id)
                frame=pd.DataFrame(dict(xf=last[0],yf=last[1],zf=last[2],dphi=np.zeros(len(pairs)),
                    hel_s=np.zeros(len(pairs)),nskipped=np.zeros(len(pairs),dtype=np.int16),dist=np.zeros(len(pairs)),donePairs=np.zeros(len(pairs),dtype=np.int8)))
                hit_arrays=[pairs.hit_id.to_numpy(),pairs.nb_hit_id.to_numpy()]
                ar.candidates.addSeeds(hit_arrays,frame.copy());br.candidates.addSeeds(hit_arrays,frame.copy())
                state(ar.candidates,br.candidates)
                with threadpool_limits(limits=1),np.errstate(invalid='ignore',divide='ignore'):
                    aa.fitTracks(ar,origin_coords=origin);ba.fitTracks(br,origin_coords=origin)
                    state(ar.candidates,br.candidates);counts['fit_cases']+=1
                    saved_a=ar.candidates.copy();saved_b=br.candidates.copy()
                    aa.findPairs(ar,k=4,for_ncross=2,origin_coords=origin)
                    ba.findPairs(br,k=4,for_ncross=2,origin_coords=origin)
                    state(ar.candidates,br.candidates);counts['double_pair_rounds']+=1
                    ar.candidates=saved_a;br.candidates=saved_b

                    if ar.candidates.n:
                        aa.chooseLikelyNextHits(ar,k=4,nmax_per_crossing=2,origin_coords=origin)
                        ba.chooseLikelyNextHits(br,k=4,nmax_per_crossing=2,origin_coords=origin)
                        state(ar.candidates,br.candidates);counts['follow_cases']+=1
                        counts['three_hit_candidates']+=int((ar.candidates.ncross>=3).sum())
                        aa.fitTracks(ar);ba.fitTracks(br)
                        for crossing_mode in [3,-1]:
                            if ar.candidates.n==0:break
                            aa.findPairs(ar,k=4,for_ncross=crossing_mode,origin_coords=origin)
                            ba.findPairs(br,k=4,for_ncross=crossing_mode,origin_coords=origin)
                            state(ar.candidates,br.candidates)
                            counts['pair_rounds']+=1
                            counts['pair_survivors']+=ar.candidates.n
                            counts['filled_pair_slots']+=int(np.count_nonzero(ar.candidates._candidates[:,1::2]))
                            # Completed schedules should not repeat for the same population.
                            saved=ar.candidates.copy()
                            aa.findPairs(ar,k=4,for_ncross=crossing_mode,origin_coords=origin)
                            state(ar.candidates,saved)

    assert counts['three_hit_candidates']>0
    paths=['sciona/trackml_pairing.py','sciona/trackml_ranking.py','sciona/trackml_seeding.py','sciona/trackml_fitting.py','sciona/trackml_extension.py',
           'sciona/trackml_candidates.py','sciona/trackml_neighbors.py','sciona/trackml_intersections.py',
           'sciona/trackml_detector.py','sciona/trackml_nearest.py','sciona/trackml_geometry.py',
           'sciona/trackml_helix.py','sciona/trackml_corrections.py','sciona/trackml_layer_functions.py',
           'sciona/trackml_bayesian.py','scripts/validate_trackml_seeding.py',
           'scripts/validate_trackml_neighbors.py','scripts/validate_trackml_detector.py',
           'scripts/validate_trackml_layer_integration.py','docs/reviews/competition_trackml_source_pins.json',
           'docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Analytic synthetic hits; first-hit query population limited to twelve for each case.',
                     'Seeding to first following step only; full findTracks scheduling, ranking and commit integration remain.',
                     'Cell-enabled extension and learned calibration remain separate gates.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_seeding.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
