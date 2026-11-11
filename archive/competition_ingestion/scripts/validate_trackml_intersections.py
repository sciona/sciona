"""Source intersection/retry parity over independently discovered synthetic specs."""
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
from sciona.trackml_intersections import CylinderIntersector,CapIntersector,Intersector
from scripts.validate_trackml_detector import synthetic_modules


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    path=source/'trackml_solution/geometry.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/geometry.py']
    names={'df_vec2_length','CylindersSpec','CapsSpec','DetectorSpec','CylinderIntersector','CapIntersector','Intersector'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==7
    ns=dict(np=np,pd=pd,KMeans=KMeans,MeanShift=MeanShift)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-trackml-intersections>','exec'),ns)
    modules=synthetic_modules(scale=10)
    with threadpool_limits(limits=1):
        actual_spec=discover_detector(modules)
        expected_spec=ns['DetectorSpec'](SimpleNamespace(detectors_df=modules))
    counts=dict(cylinder=0,cap=0,combined=0,rays=0);recursive_calls=0;hits=0;misses=0
    vector_cases=0
    def trace(obj):
        calls=[];original=obj.intersectHelices
        def call(*args,**kwargs):
            calls.append((len(args[0]),kwargs.get('iterations',10)))
            return original(*args,**kwargs)
        obj.intersectHelices=call
        return calls
    def compare(a,b):
        nonlocal hits,misses
        assert len(a)==len(b)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        index=4 if len(a)==8 else 3
        present=a[index]>=0
        hits+=int(present.sum());misses+=int((~present).sum())
        for coordinate in a[:3]:
            assert np.isfinite(coordinate[present]).all()
            assert np.isnan(coordinate[~present]).all()
        assert np.isinf(a[-2][~present]).all()
        assert (a[-1][~present]==0).all()
    with np.errstate(divide='ignore',invalid='ignore'):
        for seed in range(8):
            rng=np.random.RandomState(seed+431);n=128
            xm=rng.uniform(-300,300,n);ym=rng.uniform(-300,300,n)
            radius=rng.uniform(10,500,n);phi=rng.uniform(-np.pi,np.pi,n)
            values=[xm+radius*np.cos(phi),ym+radius*np.sin(phi),rng.uniform(-450,450,n),
                    rng.choice([-1.,1.],n),xm,ym,radius,rng.uniform(20,2000,n)*rng.choice([-1.,1.],n)]
            before=[v.copy() for v in values]
            for kind,cls,key in [('cylinder',CylinderIntersector,'cylinders'),('cap',CapIntersector,'caps')]:
                for missable in [False,True]:
                    for iterations in [1,2,10]:
                        for pre_move in [-1.,1.,15.]:
                            actual=cls(getattr(actual_spec,key));expected=ns[cls.__name__](getattr(expected_spec,key))
                            ac=trace(actual);bc=trace(expected)
                            opts=dict(iterations=iterations,pre_move=pre_move,missable=missable)
                            a=actual.intersectHelices(*values,**opts);b=expected.intersectHelices(*values,**opts)
                            compare(a,b);assert ac==bc
                            recursive_calls+=len(ac)-1;counts[kind]+=1
                            valid=a[3]>=0
                            if kind=='cylinder':
                                np.testing.assert_allclose(a[0][valid]**2+a[1][valid]**2,actual_spec.cylinders.cyl_rsqr[a[3][valid]],rtol=1e-10,atol=1e-7)
                            else:np.testing.assert_array_equal(a[2][valid],actual_spec.caps.cap_z[a[3][valid]])
            for missable in [False,True]:
                for pre_move in [None,'back',1.]:
                    for force in [None,rng.choice([False,True],n)]:
                        actual=Intersector(actual_spec);expected=ns['Intersector'](expected_spec)
                        ac=[trace(actual.cylis),trace(actual.capis)];bc=[trace(expected.cylis),trace(expected.capis)]
                        opts=dict(cyl_pre_move=pre_move,cap_pre_move=pre_move,missable=missable,force_cyl_closer=force)
                        a=actual.findNextHelixIntersection(*values,**opts);b=expected.findNextHelixIntersection(*values,**opts)
                        compare(a,b);assert ac==bc;counts['combined']+=1
                        if force is not None:np.testing.assert_array_equal(a[3],force)
            # Independent oracle: the unmodified combined selector with scalar
            # overrides for each helix. This also checks recursive default resets.
            cp=rng.uniform(-60,60,n);kp=rng.uniform(-60,60,n)
            cp_before=cp.copy();kp_before=kp.copy()
            for cyl_pre,cap_pre in [(cp,kp),(cp,'back'),(None,kp)]:
                for missable in [False,True]:
                    for force in [None,rng.choice([False,True],n)]:
                        actual=Intersector(actual_spec).findNextHelixIntersection(
                            *values,cyl_pre_move=cyl_pre,cap_pre_move=cap_pre,
                            missable=missable,force_cyl_closer=force)
                        rows=[]
                        for i in range(n):
                            rows.append(ns['Intersector'](expected_spec).findNextHelixIntersection(
                                *[v[i:i+1] for v in values],
                                cyl_pre_move=float(cyl_pre[i]) if isinstance(cyl_pre,np.ndarray) else cyl_pre,
                                cap_pre_move=float(cap_pre[i]) if isinstance(cap_pre,np.ndarray) else cap_pre,
                                missable=missable,force_cyl_closer=None if force is None else force[i:i+1]))
                        expected=tuple(np.concatenate(parts) for parts in zip(*rows))
                        compare(actual,expected);vector_cases+=1
            np.testing.assert_array_equal(cp,cp_before);np.testing.assert_array_equal(kp,kp_before)
            for a,b in zip(values,before):np.testing.assert_array_equal(a,b)
        for dimensions in [2,3]:
            starts=np.zeros((16,dimensions),dtype=np.float64)
            directions=np.random.RandomState(439).normal(size=starts.shape)
            actual=CylinderIntersector(actual_spec.cylinders).intersectFromInside(0,starts,directions)
            expected=ns['CylinderIntersector'](expected_spec.cylinders).intersectFromInside(0,starts,directions)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_allclose(np.linalg.norm(actual[:,:2],axis=1),50.,atol=1e-12)
            counts['rays']+=1
    assert recursive_calls>0 and hits>0 and misses>0
    paths=['sciona/trackml_intersections.py','sciona/trackml_detector.py','scripts/validate_trackml_intersections.py',
           'scripts/validate_trackml_detector.py','docs/reviews/competition_trackml_source_pins.json',
           'docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,exact_source_cases=counts,recursive_calls=recursive_calls,
                vector_premove_scalar_source_cases=vector_cases,
                successful_intersections=hits,missed_intersections=misses,analytic_surfaces_passed=True,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Synthetic discovered geometry and finite float64 helix inputs; no real detector records.',
                             'Original miss sentinels and recursive default pre-move behavior retained.',
                             'Corrector hooks retained but not exercised here; learned correction maps remain a separate gate.',
                             'Combined array pre-move uses guarded string dispatch; checked against unmodified source scalar calls per helix.',
                             'Intersections only; neighbor/correction/candidate lifecycle and CDG publication remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_intersections.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
