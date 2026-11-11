"""Source round setup and parameter precedence on synthetic complete-layer populations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sciona.trackml_detector import discover_detector
from sciona.trackml_intersections import Intersector
from sciona.trackml_runtime import TrackRuntime
from scripts.validate_trackml_detector import synthetic_modules
from scripts.validate_trackml_layer_integration import source_namespace, SyntheticMaps
from scripts.validate_trackml_neighbors import synthetic_hits, LegacyFrame


def validate(root,source):
    ns=source_namespace(root,source);ns['re']=re
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    for name in ['candidates.py','algorithm.py']:
        path=source/'trackml_solution'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/'+name]
        tree=ast.parse(path.read_text())
        if name=='candidates.py':nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        else:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Algorithm')
            nodes=[ast.ClassDef(name='OriginalRuntime',bases=[],keywords=[],decorator_list=[],body=[
                n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'Run','paramsForRun','setupRun'}])]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<original-'+name+'>','exec'),ns)
    modules=synthetic_modules(scale=10)
    with threadpool_limits(limits=1):
        spec=discover_detector(modules);ospec=ns['DetectorSpec'](SimpleNamespace(detectors_df=modules))
    params={'x':10.,'upto1__ADD__x':3.,'from1__SUB__x':2.,'from2__x':20.,
            'sunset__ADD__x':5.,'sunset__SUB__x':1.,'sunset__x':99.,
            'nb__cyl_scale':5.,'from1__nb__cyl_scale':2.}
    counts=dict(parameter_cases=0,setup_cases=0)
    a=TrackRuntime(spec,Intersector(spec),params)
    b=ns['OriginalRuntime']();b.spec=ospec;b.params=params.copy();b.layer_functions=None
    for commit in range(4):
        for sunset in [False,True]:
            actual=a.paramsForRun(commit,sunset);expected=b.paramsForRun(commit,sunset)
            assert actual==expected
            x=10.+(3. if commit<=1 else 0.)-(2. if commit>=1 else 0.)
            if commit>=2:x=20.
            if sunset:x=99.
            assert actual['x']==x
            assert actual['nb__cyl_scale']==(2. if commit>=1 else 5.)
            assert a.params==params and b.params==params
            counts['parameter_cases']+=1
    for seed in range(4):
        hits=synthetic_hits(modules,1931+seed)
        for maps in [None,SyntheticMaps()]:
            a.layer_functions=maps;b.layer_functions=maps
            event=SimpleNamespace(hits_df=hits,max_hit_id=len(hits),has_cells=False)
            oe=SimpleNamespace(hits_df=LegacyFrame(hits),max_hit_id=len(hits),has_cells=False)
            with threadpool_limits(limits=1):
                ar=a.setupRun(event);br=b.setupRun(oe)
            for x,y in [(ar.hit_in_cyl,br.hit_in_cyl),(ar.hit_layer_id,br.hit_layer_id)]:np.testing.assert_array_equal(x,y)
            assert ar.full_neighbors is ar.neighbors and ar.cell_features is None
            for commit in [1,2]:
                used=np.zeros(len(hits)+1,dtype=bool);used[1::(commit+2)]=True
                before=used.copy()
                with threadpool_limits(limits=1):
                    aa=a.setupRun(event,i_commit=commit,used=used,first_run=ar,sunset=commit==2)
                    bb=b.setupRun(oe,i_commit=commit,used=used,first_run=br,sunset=commit==2)
                assert aa.params==bb.params
                assert aa.available_hits_fraction==bb.available_hits_fraction==float((~used[1:]).sum())/len(hits)
                assert aa.hit_in_cyl is ar.hit_in_cyl and aa.hit_layer_id is ar.hit_layer_id
                assert aa.full_neighbors is ar.neighbors and aa.neighbors is not ar.neighbors
                assert aa.hasLayerFunction('dp0')==(maps is not None)
                all_ids=[]
                for attr,group,ids in [('cyln','cylinders','cyl_hit_id_array'),('capn','caps','cap_hit_id_array')]:
                    for al,bl in zip(getattr(getattr(aa.neighbors,attr),group),getattr(getattr(bb.neighbors,attr),group)):
                        np.testing.assert_array_equal(al.nb_coords,bl.nb_coords)
                        np.testing.assert_array_equal(getattr(al,ids),getattr(bl,ids));all_ids.extend(getattr(al,ids))
                np.testing.assert_array_equal(np.sort(all_ids),hits.hit_id.to_numpy()[~used[1:]])
                np.testing.assert_array_equal(used,before)
                counts['setup_cases']+=1
    paths=['sciona/trackml_runtime.py','sciona/trackml_neighbors.py','sciona/trackml_candidates.py',
           'sciona/trackml_layer_functions.py','sciona/trackml_detector.py',
           'scripts/validate_trackml_runtime.py','scripts/validate_trackml_neighbors.py',
           'scripts/validate_trackml_layer_integration.py','scripts/validate_trackml_detector.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['No cells in these setup cases; complete nonempty remaining layer populations only.',
                     'Full findTracks commit scheduling and cell-enabled lifecycle remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
