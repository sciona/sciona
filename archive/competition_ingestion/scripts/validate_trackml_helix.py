"""Pinned-source and analytic pitch/movement comparisons on synthetic helices."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from sciona.trackml_helix import pitch_from_two_points,pitch_least_squares,move_along_helix


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    path=source/'trackml_solution/geometry.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/geometry.py']
    names={'helixPitchFromTwoPoints','helixPitchLeastSquares','helixMove'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    ns=dict(np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-trackml-helix>','exec'),ns)
    cases={'two_point':0,'least_squares':0,'movement':0};analytic=0
    def check(label,fn,name,values,**options):
        before=[v.copy() for v in values]
        actual=fn(*values,**options);expected=ns[name](*values,**options)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        for a,b in zip(values,before):np.testing.assert_array_equal(a,b)
        cases[label]+=1
        return actual
    for seed in range(12):
        rng=np.random.RandomState(seed);n=64
        xm=rng.normal(size=n);ym=rng.normal(size=n);radius=rng.uniform(.5,10,n)
        phi=rng.uniform(-np.pi,np.pi,n);direction=rng.choice([-1.,1.],n)
        step=rng.uniform(.1,.9,n)*direction
        pitch=rng.uniform(.3,12,n)*rng.choice([-1.,1.],n);z=rng.normal(size=n)
        points=[(xm+radius*np.cos(phi+j*step),ym+radius*np.sin(phi+j*step),z+j*step*pitch/(2*np.pi)) for j in range(3)]
        two=[*points[0],*points[1],xm,ym];three=[*points[0],*points[1],*points[2],xm,ym]
        for replacement in [.001,-.02,0.]:
            a=check('two_point',pitch_from_two_points,'helixPitchFromTwoPoints',two,zero_pitch=replacement)
            b=check('least_squares',pitch_least_squares,'helixPitchLeastSquares',three,zero_pitch=replacement)
            np.testing.assert_allclose(a[0],pitch,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(b[0],pitch,rtol=1e-12,atol=1e-12)
            assert np.max(b[3])<1e-24
            analytic+=2
        zero_two=[v.copy() for v in two];zero_two[2].fill(0);zero_two[5].fill(0)
        zero_three=[v.copy() for v in three]
        for i in [2,5,8]:zero_three[i].fill(0)
        for replacement in [.001,-.02,0.]:
            a=check('two_point',pitch_from_two_points,'helixPitchFromTwoPoints',zero_two,zero_pitch=replacement)
            b=check('least_squares',pitch_least_squares,'helixPitchLeastSquares',zero_three,zero_pitch=replacement)
            np.testing.assert_array_equal(a[0],np.full(n,replacement));np.testing.assert_array_equal(b[0],np.full(n,replacement))
        distance=rng.uniform(-20,20,n);uz=rng.choice([-1.,0.,1.],n)
        values=[*points[0],uz,xm,ym,radius,pitch,distance]
        moved=check('movement',move_along_helix,'helixMove',values)
        np.testing.assert_allclose(np.hypot(moved[0]-xm,moved[1]-ym),radius,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(moved[2]-z,moved[3]*pitch/(2*np.pi),rtol=1e-13,atol=1e-13)
        returned=check('movement',move_along_helix,'helixMove',[*moved[:3],uz,xm,ym,radius,pitch,-distance])
        for a,b in zip(returned[:3],points[0]):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        analytic+=3
    # Exact +/-pi branch boundaries and identical phase in two-point routine.
    v=lambda x:np.asarray(x,dtype=np.float64)
    two=[v([1,-1,1]),v([0,0,0]),v([0,0,0]),v([-1,1,1]),v([0,0,0]),v([1,1,1]),v([0,0,0]),v([0,0,0])]
    a=check('two_point',pitch_from_two_points,'helixPitchFromTwoPoints',two)
    np.testing.assert_array_equal(a[1],v([np.pi,np.pi,.001]))
    three=[v([1]),v([0]),v([0]),v([-1]),v([0]),v([1]),v([0]),v([1]),v([2]),v([0]),v([0])]
    b=check('least_squares',pitch_least_squares,'helixPitchLeastSquares',three)
    np.testing.assert_array_equal(b[1],v([-np.pi/2]))
    for label,fn,name,arity in [('two_point',pitch_from_two_points,'helixPitchFromTwoPoints',8),
                               ('least_squares',pitch_least_squares,'helixPitchLeastSquares',11),
                               ('movement',move_along_helix,'helixMove',9)]:
        check(label,fn,name,[np.empty(0,dtype=np.float64) for _ in range(arity)])
    # Identical phases give the source least-squares calculation zero variance.
    try:pitch_least_squares(*[v([0]) for _ in range(11)])
    except ValueError:pass
    else:raise AssertionError('Undefined source regression accepted')
    paths=['sciona/trackml_helix.py','scripts/validate_trackml_helix.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,exact_source_cases=cases,analytic_checks=analytic,
                exact_half_turn_and_zero_phase_cases=True,nonfinite_regression_rejected=True,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Finite float64 aligned vectors; source nonfinite regression/movement outputs reject.',
                             'Source two-point and least-squares phase conventions intentionally differ at half-turn boundaries.',
                             'Pitch/movement components only; full TrackML reconstruction and publication remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_helix.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
