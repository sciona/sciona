"""Source and brute-force neighbor checks using analytic synthetic detector hits."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sciona.trackml_detector import discover_detector
from sciona.trackml_neighbors import Neighbors
from scripts.validate_trackml_detector import synthetic_modules
from scripts.validate_trackml_layer_integration import source_namespace, SyntheticMaps


class LegacyFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return LegacyFrame

    def as_matrix(self, columns=None):
        return self.to_numpy() if columns is None else self.loc[:,columns].to_numpy()


def synthetic_hits(modules, seed):
    rng=np.random.RandomState(seed)
    frames=[]
    for i in range(12):
        frame=modules[['volume_id','layer_id','cx','cy','cz']].rename(columns={'cx':'x','cy':'y','cz':'z'}).copy()
        angle=rng.uniform(-np.pi,np.pi,len(frame))
        radius=np.hypot(frame.x,frame.y).to_numpy()
        frame['x']=radius*np.cos(angle);frame['y']=radius*np.sin(angle)
        # Nonzero z perturbation exercises cap projection onto its reference plane.
        frame['z']=frame.z+rng.uniform(-.2,.2,len(frame))
        frames.append(frame)
    hits=pd.concat(frames,ignore_index=True)
    hits['hit_id']=np.arange(1,len(hits)+1)
    return hits


def validate(root,source):
    ns=source_namespace(root,source)
    defaults=ns['Neighbors'].default_params.copy()
    counts=dict(fits=0,direct_searches=0,brute_force_searches=0,grouped_searches=0,first_hit_sets=0)
    modules=synthetic_modules(scale=10)
    with threadpool_limits(limits=1):
        spec=discover_detector(modules)
        original_spec=ns['DetectorSpec'](SimpleNamespace(detectors_df=modules))
    def same(a,b):
        if a is None or b is None: assert a is b
        else: pd.testing.assert_frame_equal(pd.DataFrame(a),pd.DataFrame(b))
    def brute(result,layer,query,k=None,radius=None):
        distances=np.linalg.norm(query[:,None,:]-layer.nb_coords[None,:,:],axis=2)
        hit_ids=layer.cyl_hit_id_array if hasattr(layer,'cyl_hit_id_array') else layer.cap_hit_id_array
        for i in range(len(query)):
            row=result[result.vind==i].sort_values('nb_dist')
            indices=np.argsort(distances[i])[:k] if k is not None else np.flatnonzero(distances[i]<=radius)
            indices=indices[np.argsort(distances[i,indices])]
            np.testing.assert_array_equal(row.nb_hit_id.to_numpy(),hit_ids[indices])
            np.testing.assert_allclose(row.nb_dist.to_numpy(),distances[i,indices],rtol=1e-10,atol=1e-10)
        counts['brute_force_searches']+=1
    with threadpool_limits(limits=1):
        for seed in range(4):
            hits=synthetic_hits(modules,1131+seed);before=hits.copy(deep=True)
            rng=np.random.RandomState(1231+seed)
            for scale in [1.,5.]:
                for bins in [1,7]:
                    params={'nb__cyl_scale':scale,'nb__nbins_radius':bins}
                    maps=SyntheticMaps(dimensions=2)
                    a=Neighbors(spec,params=params,layer_functions=maps)
                    ns['Neighbors'].default_params=defaults.copy()
                    b=ns['Neighbors'](original_spec,params=params,layer_functions=maps)
                    ah=np.zeros(len(hits)+1,dtype=bool);bh=ah.copy()
                    al=np.full(len(hits)+1,-1,dtype=int);bl=al.copy()
                    a.fit(hits,hit_in_cyl=ah,hit_layer_id=al)
                    b.fit(LegacyFrame(hits),hit_in_cyl=bh,hit_layer_id=bl)
                    np.testing.assert_array_equal(ah,bh);np.testing.assert_array_equal(al,bl)
                    assert np.all(al[1:]>=0)
                    counts['fits']+=1
                    for family,actual,expected,layers in [('cylinder',a.cyln,b.cyln,a.cyln.cylinders),
                                                         ('cap',a.capn,b.capn,a.capn.caps)]:
                        assert len(layers)==(3 if family=='cylinder' else 4)
                        for layer_id,layer in enumerate(layers):
                            source_layer=(expected.cylinders if family=='cylinder' else expected.caps)[layer_id]
                            np.testing.assert_array_equal(layer.nb_coords,source_layer.nb_coords)
                            frame=layer.cyl_hits_df if family=='cylinder' else layer.cap_hits_df
                            query=frame[['x','y','z']].to_numpy()[:8]+rng.uniform(.1,.3,(8,3))
                            projected=layer.transformToNeighborCoords(query)
                            if family=='cylinder':
                                factor=layer.cyl_mean_r2/np.hypot(query[:,0],query[:,1])
                                analytic=query*factor[:,None]*np.array([scale,scale,1.])
                            else: analytic=query[:,:2]*(layer.cap_z/query[:,2])[:,None]
                            np.testing.assert_allclose(projected,analytic,rtol=1e-14,atol=1e-12)
                            for k in [1,5]:
                                result=actual.findNeighborhoodK(layer_id,query,k)
                                same(result,expected.findNeighborhoodK(layer_id,query,k))
                                brute(result,layer,projected,k=k);counts['direct_searches']+=1
                            for radius in [.01,30.,1000.]:
                                result=actual.findNeighborhood(layer_id,query,radius)
                                same(result,expected.findNeighborhood(layer_id,query,radius))
                                brute(result,layer,projected,radius=radius);counts['direct_searches']+=1
                    # Mixed layers, nonfinite query rejection, variable-radius binning
                    # and restoration of original query indices after grouping.
                    chosen=rng.choice(len(hits),48,replace=False)
                    xyz=hits.iloc[chosen][['x','y','z']].to_numpy().T.copy()
                    xyz+=rng.uniform(.1,.3,xyz.shape);xyz[:,0]=np.nan
                    hit_ids=hits.iloc[chosen].hit_id.to_numpy()
                    family=ah[hit_ids];ids=al[hit_ids]
                    radii=rng.uniform(.01,100.,48)
                    for k in [1,5]:
                        same(a.findIntersectionNeighborhoodK(*xyz,family,ids,k),
                             b.findIntersectionNeighborhoodK(*xyz,family,ids,k))
                        counts['grouped_searches']+=1
                    same(a.findIntersectionNeighborhood(*xyz,family,ids,radii),
                         b.findIntersectionNeighborhood(*xyz,family,ids,radii))
                    counts['grouped_searches']+=1
                    xyz[:]=np.nan
                    assert a.findIntersectionNeighborhood(*xyz,family,ids,radii) is None
                    assert b.findIntersectionNeighborhood(*xyz,family,ids,radii) is None
                    same(a.findFirstHitNeighborhood(),b.findFirstHitNeighborhood())
                    counts['first_hit_sets']+=1
            pd.testing.assert_frame_equal(hits,before)
    # Local compatibility fix: independent instances do not mutate class defaults.
    isolated=Neighbors(spec,params={'nb__cyl_scale':19.})
    other=Neighbors(spec)
    assert isolated.params['nb__cyl_scale']==19. and other.params['nb__cyl_scale']==5.
    assert Neighbors.default_params==defaults
    paths=['sciona/trackml_neighbors.py','sciona/trackml_layer_functions.py','sciona/trackml_detector.py',
           'scripts/validate_trackml_neighbors.py','scripts/validate_trackml_layer_integration.py',
           'scripts/validate_trackml_detector.py','docs/reviews/competition_trackml_source_pins.json',
           'docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Synthetic complete layer populations and one fit per instance; sparse layers/repeated fit retain source assumptions.',
                     'Legacy pandas column extraction adapted; source oracle uses a local DataFrame compatibility subclass.',
                     'Instance defaults copied; source oracle starts from fresh defaults per fit.',
                     'Radius bins retain source over-selection; downstream candidate filters remain to validate.',
                     'Bayesian scoring, candidate lifecycle and CDG publication remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_neighbors.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
