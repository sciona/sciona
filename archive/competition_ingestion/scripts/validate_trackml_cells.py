"""Source cell-feature and candidate-filter parity on analytic synthetic fixtures."""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sciona.trackml_cells import CellFeatures
from sciona.trackml_extension import TrackExtension
from scripts.validate_trackml_neighbors import LegacyFrame


class LegacySeries(pd.Series):
    def __getitem__(self,key):
        if isinstance(key,tuple):return self.to_numpy()[key]
        return super().__getitem__(key)
    @property
    def _constructor(self):return LegacySeries

class LegacyFilterFrame(pd.DataFrame):
    @property
    def _constructor(self):return LegacyFilterFrame
    @property
    def _constructor_sliced(self):return LegacySeries


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    ns=dict(np=np)
    for filename in ['cells.py','geometry.py','algorithm.py']:
        path=source/'trackml_solution'/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/'+filename]
        tree=ast.parse(path.read_text())
        if filename=='cells.py':nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        elif filename=='geometry.py':nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='helixUnitTangentVector']
        else:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Algorithm')
            method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='dropNeighborsInconsistentWithCellFeatures')
            nodes=[ast.ClassDef(name='OriginalFilter',bases=[],keywords=[],body=[method],decorator_list=[])]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<original-'+filename+'>','exec'),ns)
    counts=dict(feature_cases=0,direction_cases=0,filter_cases=0,kept=0,rejected=0)
    for seed in range(8):
        rng=np.random.RandomState(1731+seed)
        modules=[];rotations=[]
        for i in range(3):
            rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(rotation)<0:rotation[:,0]*=-1
            rotations.append(rotation)
            row=dict(volume_id=1,layer_id=1,module_id=i+1,pitch_u=.1*(i+1),pitch_v=.2,module_t=.3)
            row.update({f'rot_{dst}{src}':rotation[j,k] for j,dst in enumerate('xyz') for k,src in enumerate('uvw')})
            modules.append(row)
        modules=pd.DataFrame(modules)
        patterns=[[(0,0)],[(0,0),(0,0)],[(0,0),(1,0),(3,0)],
                  [(0,0),(0,2),(0,3)],[(0,0),(1,2),(2,4)],[(0,3),(1,2),(3,0)]]
        rows=[];hits=[]
        for i in range(18):
            hits.append(dict(hit_id=i+1,volume_id=1,layer_id=1,module_id=i%3+1))
            for u,v in patterns[i%6]:rows.append(dict(hit_id=i+1,ch0=u,ch1=v,value=float(rng.uniform(.1,2.))))
        hits=pd.DataFrame(hits);cells=pd.DataFrame(rows)
        for shuffled in [False,True]:
            c=cells.sample(frac=1,random_state=seed).reset_index(drop=True) if shuffled else cells.copy()
            event=SimpleNamespace(has_cells=True,cells_df=c,hits_df=hits,max_hit_id=18)
            def algo(frame):return SimpleNamespace(spec=SimpleNamespace(geospec=SimpleNamespace(detectors_df=frame)),
                                                   timed=lambda *args:nullcontext(),log=lambda *args:None)
            ad=SimpleNamespace();bd=SimpleNamespace()
            a=CellFeatures(algo(modules),SimpleNamespace(event=event),details=ad)
            b=ns['CellFeatures'](algo(LegacyFrame(modules)),SimpleNamespace(event=event),details=bd)
            for x,y in zip(a._cell_features,b._cell_features):np.testing.assert_array_equal(x,y)
            for key in vars(ad):np.testing.assert_array_equal(getattr(ad,key),getattr(bd,key))
            for vec in a._cell_features[:2]:np.testing.assert_allclose(np.linalg.norm(vec,axis=1),1.,atol=1e-14)
            expected_counts=np.bincount(c.hit_id,minlength=19)
            np.testing.assert_array_equal(a._cell_features[2],expected_counts)
            for i in range(18):
                if i%6 in [0,1]:
                    np.testing.assert_allclose(a._cell_features[0][i+1],rotations[i%3][:,2],atol=1e-14)
                    assert ad.hl2_by_hit[i+1]==0
            counts['feature_cases']+=1
            ids=np.arange(1,19)
            for kind in ['random','up','reverse','scaled']:
                query=rng.normal(size=(18,3));query/=np.linalg.norm(query,axis=1)[:,None]
                if kind=='up':query=a._cell_features[0][ids].copy()
                if kind=='reverse':query=-a._cell_features[1][ids].copy()
                if kind=='scaled':query*=2.
                actual=a.estimateClosestDirection(ids,query);expected=b.estimateClosestDirection(ids,query)
                for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
                alternatives=np.stack([a._cell_features[0][ids],-a._cell_features[0][ids],
                                       a._cell_features[1][ids],-a._cell_features[1][ids]],axis=1)
                best=np.max(np.sum(alternatives*query[:,None,:],axis=2),axis=1)
                np.testing.assert_allclose(actual[1],best,atol=1e-14)
                np.testing.assert_allclose(actual[2],np.sqrt(expected_counts[ids])*(1-best),atol=1e-14)
                counts['direction_cases']+=1
            xyz=rng.uniform(10,50,(3,19))
            event.hitCoordinatesById=lambda h:tuple(xyz[:,h].copy())
            frame=pd.DataFrame(dict(extend_hit_id=ids,hel_dz=rng.choice([-1.,1.],18),
                hel_xm=np.zeros(18),hel_ym=np.zeros(18),hel_r=np.full(18,50.),hel_pitch=np.full(18,100.)))
            in_cyl=np.arange(19)%2==0;layers=np.arange(19)%6
            for cut in [0.,.5,10.]:
                ar=SimpleNamespace(event=event,cell_features=a,hit_in_cyl=in_cyl,hit_layer_id=layers,params={'nb__cells_cut':cut})
                br=SimpleNamespace(event=event,cell_features=b,hit_in_cyl=in_cyl,hit_layer_id=layers,params={'nb__cells_cut':cut})
                actual=TrackExtension(None,None).dropNeighborsInconsistentWithCellFeatures(ar,frame)
                expected=ns['OriginalFilter']().dropNeighborsInconsistentWithCellFeatures(br,LegacyFilterFrame(frame))
                pd.testing.assert_frame_equal(actual,pd.DataFrame(expected))
                unfiltered=~(in_cyl[ids]&(layers[ids]<4))
                assert set(ids[unfiltered]).issubset(set(actual.extend_hit_id))
                counts['filter_cases']+=1;counts['kept']+=len(actual);counts['rejected']+=len(frame)-len(actual)
    assert counts['rejected']>0
    paths=['sciona/trackml_cells.py','sciona/trackml_extension.py','sciona/trackml_nearest.py',
           'scripts/validate_trackml_cells.py','scripts/validate_trackml_neighbors.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Synthetic positive cell weights, complete per-hit cells and orthonormal module rotations only.',
                     'Source repeated-single-cell behavior retained; no physical calibration or real detector claim.',
                     'Full extension with cells, repeated fit/follow, seeding and tracking lifecycle remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_cells.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
