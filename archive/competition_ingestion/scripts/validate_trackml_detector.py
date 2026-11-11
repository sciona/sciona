"""Original detector discovery on wholly synthetic module layouts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,MeanShift
from threadpoolctl import threadpool_limits
from sciona.trackml_detector import discover_detector


def synthetic_modules(scale=1.,paired=False,dtype=np.float64):
    rows=[]
    # Software volume classifications, synthetic positions only. These layouts
    # are constructed analytically and contain no original detector records.
    for layer,(radius,extent) in enumerate([(5.,10.),(15.,20.),(25.,30.)],1):
        for sign in [-1,1]:rows.append(dict(volume_id=8,layer_id=layer,cx=sign*radius*scale,cy=0.,cz=sign*extent*scale,module_hv=2.*scale))
    for z in [-40.,-20.,20.,40.]:
        for offset in ([0.,.5] if paired else [0.]):
            volume=(7 if z<0 else 9)+(5 if offset and z<0 else 5 if offset else 0)
            for radius in [10.,14.,30.,34.,50.,54.,70.]:
                rows.append(dict(volume_id=volume,layer_id=int(abs(z)/20),cx=radius*scale,cy=0.,cz=z*scale+offset,module_hv=3.*scale))
    frame=pd.DataFrame(rows)
    for name in ['cx','cy','cz','module_hv']:frame[name]=frame[name].astype(dtype)
    return frame


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    path=source/'trackml_solution/geometry.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/geometry.py']
    names={'df_vec2_length','CylindersSpec','CapsSpec','DetectorSpec'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==4
    ns=dict(np=np,pd=pd,KMeans=KMeans,MeanShift=MeanShift)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-trackml-detector>','exec'),ns)
    cases=0;arrays=0;frames=0
    with threadpool_limits(limits=1):
        for dtype in [np.float32,np.float64]:
            for scale in [.8,1.,1.3]:
                for paired in [False,True]:
                    for seed in [0,1]:
                        modules=synthetic_modules(scale,paired,dtype).sample(frac=1,random_state=seed).reset_index(drop=True)
                        before=modules.copy(deep=True)
                        expected=ns['DetectorSpec'](SimpleNamespace(detectors_df=modules))
                        actual=discover_detector(modules)
                        assert len(actual.cylinders)==len(expected.cylinders)==3
                        assert len(actual.caps)==len(expected.caps)==4
                        for a,b in [(actual.cylinders,expected.cylinders),(actual.caps,expected.caps)]:
                            assert vars(a).keys()==vars(b).keys()
                            for name,value in vars(a).items():
                                reference=getattr(b,name)
                                if isinstance(value,pd.DataFrame):pd.testing.assert_frame_equal(value,reference,check_exact=True);frames+=1
                                else:np.testing.assert_array_equal(value,reference);arrays+=1
                        np.testing.assert_allclose(actual.cylinders.cyl_rsqr,np.square(np.array([5.,15.,25.])*scale),rtol=1e-6)
                        np.testing.assert_allclose(actual.cylinders.cyl_absz_max,np.array([12.,22.,32.])*scale,rtol=1e-6)
                        assert len(actual.caps.ring_gap_r2_min)==3
                        assert np.all(np.diff(actual.caps.cap_z)>0)
                        pd.testing.assert_frame_equal(modules,before,check_exact=True)
                        cases+=1
        invalid=synthetic_modules()
        invalid.loc[(invalid.volume_id==8)&(invalid.layer_id==1),'module_hv']=100.
        rejected=0
        for fn in [lambda:discover_detector(invalid),lambda:ns['DetectorSpec'](SimpleNamespace(detectors_df=invalid))]:
            try:fn()
            except AssertionError:rejected+=1
        assert rejected==2
    paths=['sciona/trackml_detector.py','scripts/validate_trackml_detector.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,exact_detector_cases=cases,exact_array_checks=arrays,
                exact_frame_checks=frames,source_monotonicity_rejection=True,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Analytic synthetic module layouts only; original geometry files not used.',
                             'Current sklearn KMeans/MeanShift and pandas backend; historical dependency binaries excluded.',
                             'Original fixed seven-ring/three-gap assumptions retained; not a generic detector discovery algorithm.',
                             'Detector-discovery component only; intersections, corrections and full tracking lifecycle remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_detector.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
