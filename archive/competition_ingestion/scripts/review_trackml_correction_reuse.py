"""Compare reusable correction atoms to pinned source on synthetic numeric inputs."""
import ast
import inspect
import json
from types import SimpleNamespace
import numpy as np
from sciona.atoms.particle_tracking.detector_corrections.atoms import perturbative_cap_correction, perturbative_cylinder_correction
from sciona.trackml_corrections import HelixCorrector
from scripts.audit_decomposed_trackml_readiness import ROOT, sha


def source_class():
    pins=json.loads((ROOT/'docs/reviews/competition_trackml_source_pins.json').read_text())
    path=ROOT.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking/solution_trackml_solution_corrections.py.source'
    if sha(path)!=pins['files']['trackml_solution/corrections.py']:raise ValueError('Pinned corrections changed')
    node,=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HelixCorrector']
    namespace=dict(np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-corrections>','exec'),namespace)
    return namespace['HelixCorrector'],path


def context(x,y,z,radius,pitch,direction,first,second):
    # Deterministic caller-provided coefficients; no learned maps or detector data.
    n=len(x)
    neighbors=SimpleNamespace(evaluateLayerFunctions=lambda *a,**kw:(first.copy(),second.copy()))
    return SimpleNamespace(
        _run=SimpleNamespace(layer_functions=SimpleNamespace(functions=('dp0','dp1')),neighbors=neighbors),
        _algo=SimpleNamespace(spec=SimpleNamespace(
            caps=SimpleNamespace(caps_r2_max=np.array([np.inf]),caps_r2_min=np.array([-np.inf])),
            cylinders=SimpleNamespace(cyl_absz_max=np.array([np.inf])))),
        _hel_points=[(x.copy(),y.copy(),z.copy())],
        helixParams=lambda:(direction,np.zeros(n),np.zeros(n),radius,pitch))


def review():
    source,path=source_class()
    paths=dict(source=path,reviewer=__file__,retained=ROOT/'sciona/trackml_corrections.py',
               provider=inspect.getfile(inspect.unwrap(perturbative_cap_correction)))
    hashes={k:sha(v) for k,v in paths.items()}
    counts=dict(cap=0,cylinder=0)
    for seed in range(12):
        rng=np.random.RandomState(seed+1743)
        for scale in (.01,1.,100.):
            n=32
            x,y,z=rng.normal(size=(3,n))*scale
            radius=rng.uniform(.2,4.,n)*scale
            pitch=rng.uniform(.3,9.,n)*rng.choice([-1.,1.],n)*scale
            first,second=rng.normal(size=(2,n))*.01*scale**2
            for sign in (-1.,1.):
                direction=np.full(n,sign)
                charge=-np.sign(direction*pitch)
                state=context(x,y,z,radius,pitch,direction,first,second)
                layer=np.zeros(n,dtype=np.int8)
                for name in counts:
                    method='correctCapIntersections' if name=='cap' else 'correctCylinderIntersections'
                    extra={} if name=='cap' else dict(mask=np.ones(n,dtype=bool))
                    expected=getattr(source,method)(state,x.copy(),y.copy(),z.copy(),layer,**extra)
                    retained=getattr(HelixCorrector,method)(state,x.copy(),y.copy(),z.copy(),layer,**extra)
                    if name=='cap': actual=perturbative_cap_correction(x,y,pitch,charge,first,second)
                    else: actual=perturbative_cylinder_correction(x,y,z,radius,pitch,charge,first,second)
                    for a,b,c in zip(actual,expected,retained,strict=True):
                        np.testing.assert_array_equal(a,b)
                        np.testing.assert_array_equal(b,c)
                    counts[name]+=1
    # Existing providers substitute denominators at the axis; source is undefined.
    one=np.ones(1);zero=np.zeros(1);layer=np.zeros(1,dtype=np.int8)
    state=context(zero,zero,zero,one,one,one,one,one)
    differences=[]
    with np.errstate(invalid='ignore',divide='ignore'):
        for name in counts:
            if name=='cap':
                expected=source.correctCapIntersections(state,zero.copy(),zero.copy(),zero.copy(),layer)
                actual=perturbative_cap_correction(zero,zero,one,-one,one,one)
            else:
                expected=source.correctCylinderIntersections(state,zero.copy(),zero.copy(),zero.copy(),layer,mask=np.ones(1,dtype=bool))
                actual=perturbative_cylinder_correction(zero,zero,zero,one,one,-one,one,one)
            if all(np.isfinite(v).all() for v in expected) or not all(np.isfinite(v).all() for v in actual):
                raise AssertionError('Expected axis denominator substitution difference')
            differences.append(name+'_axis_substitution')
    if hashes!={k:sha(v) for k,v in paths.items()}:raise ValueError('Reviewed source changed')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        exact_source_batches=counts,batch_size=32,undefined_domain_differences=differences,
        implementation_sha256=hashes,
        decision='Reuse both existing correction atoms on finite nonempty aligned vectors with nonzero transverse radius and finite positive scaling denominators. Keep coefficient lookup, missing-map policy, geometry bounds and in-place state updates in the recipe; explicitly handle empty/masked batches.',
        limitations=['Existing safe-denominator substitutions are not qualified as source-equivalent; recipe boundaries must exclude undefined geometry.',
            'Charge sign is an explicit signed orientation multiplier; particle interpretation belongs to the recipe.',
            'Coefficients are caller supplied and have squared-length units for length-valued corrections; no learned calibration is bundled or qualified.',
            'Source mutates intersection arrays; pure providers return new arrays. Recipe adapters must preserve intended state updates.',
            'Synthetic ordinary-range parity does not establish arbitrary-input or historical tracking accuracy.',
            'Lifecycle integration, catalog identity checks and semantic publication remain pending.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/trackml_correction_reuse.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
