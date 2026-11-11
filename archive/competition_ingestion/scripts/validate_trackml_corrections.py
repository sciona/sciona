"""Compare correction state and formulas with pinned source using synthetic inputs."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sciona.trackml_corrections import HelixCorrector, submask


def validate(root, source):
    pins = json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    ns = dict(np=np, sys=sys)
    for name, selected in [
        ('geometry.py', {'circleFromThreePoints', 'helixPitchFromTwoPoints'}),
        ('corrections.py', {'submask', 'HelixCorrector'}),
    ]:
        path = source/'trackml_solution'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins['files']['trackml_solution/'+name]
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
        assert len(nodes) == len(selected)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original-'+name+'>', 'exec'), ns)
    counts = dict(state_updates=0, displacement_cases=0, analytic_displacements=0, submasks=0)

    def same(a, b):
        for x, y in zip(a.helixParams(), b.helixParams()):
            np.testing.assert_array_equal(x, y)
        for pa, pb in zip(a._hel_points, b._hel_points):
            for x, y in zip(pa, pb):
                np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(a._mask, b._mask)
        np.testing.assert_array_equal(a._hel_params_mask, b._hel_params_mask)

    for seed in range(8):
        rng = np.random.RandomState(831+seed)
        n = 32
        xm = rng.uniform(10, 30, n); ym = rng.uniform(-30, -10, n)
        radius = rng.uniform(40, 100, n)
        pitch = rng.uniform(100, 500, n)*rng.choice([-1., 1.], n)
        phase = rng.uniform(-1, 1, n)
        def point(offset):
            angle = phase+offset
            return (xm+radius*np.cos(angle), ym+radius*np.sin(angle), pitch*angle/(2*np.pi))
        initial = (pitch*.1/(2*np.pi), xm, ym, radius, pitch)
        spec = SimpleNamespace(
            caps=SimpleNamespace(caps_r2_min=np.array([20.]), caps_r2_max=np.array([150.])),
            cylinders=SimpleNamespace(cyl_absz_max=np.array([200.])))
        def build(cls, mode='nonzero'):
            calls = []
            def evaluate(*args, functions, points):
                assert functions == ('dp0', 'dp1')
                calls.append((args[3], args[4].copy(), points.copy()))
                d0 = np.full(len(points), 2.); d1 = np.full(len(points), -3.)
                if mode == 'nan':
                    d0[::2] = np.nan; d1[1::2] = np.nan
                return d0, d1
            algo = SimpleNamespace(spec=spec, getHelixParams=lambda *args, mask: tuple(
                p.copy() if mask is None else p[mask].copy() for p in initial))
            candidates = SimpleNamespace(hitCoordinates=lambda i, mask: tuple(
                p.copy() if mask is None else p[mask].copy() for p in point(.1 if i == -1 else 0.)))
            run = SimpleNamespace(candidates=candidates,
                layer_functions=None if mode == 'disabled' else SimpleNamespace(functions=('dp0', 'dp1')),
                neighbors=SimpleNamespace(evaluateLayerFunctions=evaluate))
            return cls(algo, run), calls

        a, _ = build(HelixCorrector); b, _ = build(ns['HelixCorrector'])
        same(a, b)
        # Clone updates stay isolated until explicit merge.
        ca = a.clone(); cb = b.clone()
        old = tuple(p.copy() for p in a.helixParams())
        ca.updateHelices(*point(.3)); cb.updateHelices(*point(.3)); same(ca, cb)
        for x, y in zip(old, a.helixParams()): np.testing.assert_array_equal(x, y)
        mask = rng.choice([False, True], n)
        a.merge(ca, mask); b.merge(cb, mask); same(a, b)
        for x, original, changed in zip(a.helixParams(), old, ca.helixParams()):
            np.testing.assert_array_equal(x, np.where(mask, changed, original))
        counts['state_updates'] += 2
        # Nested subsets share the full parameter storage; updates must propagate.
        sa = a.subset(mask); sb = b.subset(mask)
        inner = rng.choice([False, True], int(mask.sum()))
        ta = sa.subset(inner); tb = sb.subset(inner)
        selected = np.flatnonzero(mask)[inner]
        for x, y in zip(ta._hel_params, a._hel_params): assert np.shares_memory(x, y)
        ta.updateHelices(*[p[selected] for p in point(.5)])
        tb.updateHelices(*[p[selected] for p in point(.5)])
        same(ta, tb); same(sa, sb); same(a, b)
        counts['state_updates'] += 1
        # Masked full updates preserve all unselected parameter entries.
        a, _ = build(HelixCorrector); b, _ = build(ns['HelixCorrector'])
        old = tuple(p.copy() for p in a.helixParams())
        a.updateHelices(*point(.4), mask=mask); b.updateHelices(*point(.4), mask=mask)
        same(a, b)
        for x, y in zip(a.helixParams(), old): np.testing.assert_array_equal(x[~mask], y[~mask])
        counts['state_updates'] += 1
        for outer, child in [(None,None),(None,mask),(mask,None),(mask,inner)]:
            np.testing.assert_array_equal(submask(outer,child), ns['submask'](outer,child))
            counts['submasks'] += 1

        for kind in ['cap', 'cylinder']:
            for mode in ['disabled', 'nan', 'nonzero']:
                a, ac = build(HelixCorrector, mode); b, bc = build(ns['HelixCorrector'], mode)
                angle = rng.uniform(-np.pi,np.pi,n)
                r = np.linspace(5.,180.,n) if kind == 'cap' else np.full(n,80.)
                z = np.linspace(-240.,240.,n)
                coords = (r*np.cos(angle),r*np.sin(angle),z)
                ids = np.zeros(n,dtype=int)
                if kind == 'cap':
                    actual=a.correctCapIntersections(*[p.copy() for p in coords],ids)
                    expected=b.correctCapIntersections(*[p.copy() for p in coords],ids)
                else:
                    actual=a.correctCylinderIntersections(*[p.copy() for p in coords],ids,mask)
                    expected=b.correctCylinderIntersections(*[p.copy() for p in coords],ids,mask)
                for x,y in zip(actual,expected): np.testing.assert_array_equal(x,y)
                same(a,b)
                assert len(ac)==len(bc)
                for x,y in zip(ac,bc):
                    assert x[0]==y[0]
                    np.testing.assert_array_equal(x[1],y[1]);np.testing.assert_array_equal(x[2],y[2])
                counts['displacement_cases'] += 1
                if mode == 'nonzero':
                    # Closed-form checks independent of source coordinate implementation.
                    q=-np.sign(initial[0]*pitch); p=np.abs(pitch/(2*np.pi))
                    if kind == 'cap':
                        factor=q/p; factor[(r<10.) | (r>160.)]=0
                        oracle=(coords[0]+(2*np.cos(angle)+3*np.sin(angle))*factor,
                                coords[1]+(2*np.sin(angle)-3*np.cos(angle))*factor)
                    else:
                        factor=q/(radius+p*p/radius);factor[np.abs(z)>210.]=0
                        oracle=(coords[0]+3*np.sin(angle)*factor,
                                coords[1]-3*np.cos(angle)*factor,z+2*factor)
                    for x,y in zip(actual,oracle):np.testing.assert_allclose(x,y,rtol=1e-13,atol=1e-13)
                    counts['analytic_displacements'] += 1
    paths=['sciona/trackml_corrections.py','sciona/trackml_geometry.py','sciona/trackml_helix.py',
           'scripts/validate_trackml_corrections.py','docs/reviews/competition_trackml_source_pins.json',
           'docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Synthetic candidate context and displacement callbacks, not learned map interpolation.',
                     'Correction state and formulas only; intersection integration and full lifecycle remain.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_corrections.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
