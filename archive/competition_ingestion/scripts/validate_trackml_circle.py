"""Independent pinned-source and analytic checks for synthetic circle fits."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import argparse
import numpy as np
from sciona.trackml_geometry import circle_from_three_points


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    path=source/'trackml_solution/geometry.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['trackml_solution/geometry.py']
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='circleFromThreePoints']
    assert len(nodes)==1
    ns=dict(np=np,sys=sys)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-trackml-circle>','exec'),ns)
    cases=[]
    for seed in range(8):
        values=np.random.RandomState(seed).normal(size=(6,128)).astype(np.float64)
        for radius in [1.,10.,1e6]:cases.append((values,radius))
    # Collinear, repeated pairs and non-origin identical triples exercise the
    # original tangent-circle fallback without replacing its conventions.
    degenerate=np.array([[1,1,1,2],[0,2,2,3],[2,1,4,2],[0,2,5,3],[3,4,1,2],[0,5,2,3]],dtype=np.float64)
    for radius in [1.,10.,1e6]:cases.append((degenerate,radius))
    cases.append((np.zeros((6,0),dtype=np.float64),1e6))
    fits=0;diagnostic_cases=0
    for values,radius in cases:
        before=values.copy();a=io.StringIO();b=io.StringIO()
        with contextlib.redirect_stderr(a):actual=circle_from_three_points(*values,large_radius=radius)
        with contextlib.redirect_stderr(b):expected=ns['circleFromThreePoints'](*values,large_radius=np.float64(radius))
        for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
        assert a.getvalue()==b.getvalue()
        np.testing.assert_array_equal(values,before)
        fits+=values.shape[1];diagnostic_cases+=bool(a.getvalue())
    angles=np.array([0.,2.,4.]);center=np.array([3.,-2.]);radius=5.
    points=center[:,None]+radius*np.array([np.cos(angles),np.sin(angles)])
    result=circle_from_three_points(*[np.array([points[row,col]]) for col in range(3) for row in range(2)])
    np.testing.assert_allclose(np.array(result).ravel(),[3.,-2.,5.],rtol=1e-14,atol=1e-14)
    with contextlib.redirect_stderr(io.StringIO()):
        try:circle_from_three_points(*np.zeros((6,1),dtype=np.float64))
        except ValueError:pass
        else:raise AssertionError('Undefined origin tangent accepted')
    paths=['sciona/trackml_geometry.py','scripts/validate_trackml_circle.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,exact_source_cases=len(cases),point_triples=fits,
                exact_source_diagnostic_cases=diagnostic_cases,analytic_circle_passed=True,singular_origin_rejected=True,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Finite float64 vectors; source nonfinite degenerate output rejects explicitly.',
                             'Circle-fit component only; full detector geometry, neighbor search, helix corrections and track selection remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_circle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
