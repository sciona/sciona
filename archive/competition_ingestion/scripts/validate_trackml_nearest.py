"""Pinned-source and analytic synthetic nearest-point/tangent comparisons."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from sciona.trackml_nearest import (nearest_point_distance,unit_tangent,
                                   helix_from_tangent,direction_from_two_points)


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    path=source/'trackml_solution/geometry.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/geometry.py']
    functions={'nearest':(nearest_point_distance,'helixNearestPointDistance'),
               'tangent':(unit_tangent,'helixUnitTangentVector'),
               'construct':(helix_from_tangent,'helixWithTangentVector'),
               'direction':(direction_from_two_points,'helixDirectionFromTwoPoints')}
    names={pair[1] for pair in functions.values()}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==4
    ns=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-trackml-nearest>','exec'),ns)
    counts={key:0 for key in functions};analytic=0
    def check(key,args,**options):
        before=[np.asarray(a).copy() for a in args]
        fn,name=functions[key]
        actual=fn(*args,**options);expected=ns[name](*args,**options)
        if isinstance(actual,tuple):
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        else:np.testing.assert_array_equal(actual,expected)
        for a,b in zip(args,before):np.testing.assert_array_equal(a,b)
        counts[key]+=1
        return actual
    for seed in range(10):
        rng=np.random.RandomState(seed+421);n=32
        xm=rng.normal(size=n);ym=rng.normal(size=n);r=rng.uniform(.5,5,n)
        pitch=rng.uniform(1,20,n)*rng.choice([-1.,1.],n);phi=rng.uniform(-np.pi,np.pi,n)
        x0=xm+r*np.cos(phi);y0=ym+r*np.sin(phi);z0=rng.normal(size=n)
        targets=[rng.normal(size=n)*5 for _ in range(3)]
        params=[x0,y0,z0,xm,ym,r,pitch]
        for scalar in [False,True]:
            values=[float(v[0]) for v in params] if scalar else params
            for iterations in [1,3,6]:
                for extra in [False,True]:check('nearest',[*values,*targets],iterations=iterations,return_hel_s=extra)
        # On-helix targets, including multiple turns, have known zero distance.
        delta=rng.uniform(-4*np.pi,4*np.pi,n)
        target=[xm+r*np.cos(phi+delta),ym+r*np.sin(phi+delta),z0+pitch*delta/(2*np.pi)]
        result=check('nearest',[*params,*target],return_hel_s=True)
        np.testing.assert_allclose(result[3],0,atol=1e-12)
        np.testing.assert_allclose(result[4],delta,atol=1e-12)
        np.testing.assert_allclose(result[5],np.sqrt(r*r+(pitch/(2*np.pi))**2)*np.abs(delta),atol=1e-12)
        uz=rng.choice([-1.,1.],n)
        tangent=check('tangent',[x0,y0,z0,uz,xm,ym,r,pitch])
        np.testing.assert_allclose(np.linalg.norm(tangent,axis=1),1,atol=1e-14)
        np.testing.assert_array_equal(np.sign(tangent[:,2]),uz)
        restored=check('construct',[x0,y0,z0,*tangent.T,pitch])
        for a,b in zip(restored,[xm,ym,r]):np.testing.assert_allclose(a,b,atol=1e-12)
        check('construct',[x0,y0,z0,*tangent.T,float(pitch[0])],max_mg_over_qB=.1)
        for limit in [.001,.1]:
            zero_uz=uz.copy();zero_uz[::2]=0
            check('construct',[x0,y0,z0,*tangent[:,:2].T,zero_uz,pitch],min_uz0=limit)
        other=[xm+r*np.cos(phi+.2),ym+r*np.sin(phi+.2),z0+pitch*.2/(2*np.pi)]
        direction=check('direction',[xm,ym,pitch,x0,y0,z0,*other])
        np.testing.assert_array_equal(np.sign(direction),np.sign(pitch))
        flat=check('direction',[xm,ym,pitch,x0,y0,z0,other[0],other[1],z0])
        np.testing.assert_array_equal(np.sign(flat),np.sign(pitch))
        analytic+=9
    for key,size in [('nearest',10),('tangent',8),('construct',7),('direction',9)]:
        check(key,[np.empty(0,dtype=np.float64) for _ in range(size)])
    zero=np.zeros(1,dtype=np.float64);one=np.ones(1,dtype=np.float64)
    for fn,args in [(nearest_point_distance,[one,zero,zero,zero,zero,one,zero,one,zero,zero]),
                    (unit_tangent,[one,zero,zero,zero,zero,zero,one,one])]:
        try:fn(*args)
        except ValueError:pass
        else:raise AssertionError('Nonfinite source result accepted')
    paths=['sciona/trackml_nearest.py','sciona/trackml_helix.py','scripts/validate_trackml_nearest.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,exact_source_cases=counts,analytic_checks=analytic,
                scalar_parameters_and_optional_outputs=True,nonfinite_rejections=2,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Source fixed Newton iteration approximation retained; arbitrary targets are not claimed globally optimal.',
                             'Finite float64 runtime vectors and documented scalar helix parameters; nonfinite source outputs reject.',
                             'Geometry components only; detector intersections and complete TrackML lifecycle remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_nearest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
