"""Pinned-source interpolation and corrected-intersection integration on synthetic maps."""
import argparse
import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import KMeans, MeanShift
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from sciona.trackml_corrections import HelixCorrector
from sciona.trackml_detector import discover_detector
from sciona.trackml_intersections import Intersector
from sciona.trackml_layer_functions import build_layer_evaluator, createLayerFunctionInterpolators
from scripts.validate_trackml_detector import synthetic_modules


class SyntheticMaps:
    functions = ('dp0', 'dp1', 'absent')

    def __init__(self, dimensions=3, dropped=0, missing=False):
        self.dimensions = dimensions
        self.dropped = dropped
        self.missing = missing

    def formula(self, points, cylinder, layer, function):
        points = points[:, self.dropped:]
        coefficients = np.arange(1, points.shape[1]+1)*.0001
        sign = 1 if function == 'dp0' else -1
        return sign*(points@coefficients+.2+.01*layer+.02*int(cylinder))

    def getGridValues(self, *, is_cylinder, layer_id, functions):
        result = []
        for function in functions:
            if function == 'absent' or (self.missing and layer_id == 1):
                result.append((None, None)); continue
            axes = [np.array([0.]) if i < self.dropped else np.array([-100., 0., 100.])
                    for i in range(self.dimensions)]
            mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
            values = self.formula(mesh.reshape(-1, self.dimensions), is_cylinder, layer_id, function)
            result.append((axes, values.reshape(mesh.shape[:-1])))
        return result


def source_namespace(root, source):
    pins = json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    ns = dict(np=np, pd=pd, sys=sys, OrderedDict=OrderedDict,
              KMeans=KMeans, MeanShift=MeanShift, NearestNeighbors=NearestNeighbors,
              RegularGridInterpolator=RegularGridInterpolator)
    selections = {
        'geometry.py': {'df_vec2_length','CylindersSpec','CapsSpec','DetectorSpec',
                        'CylinderIntersector','CapIntersector','Intersector',
                        'circleFromThreePoints','helixPitchFromTwoPoints'},
        'neighbors.py': {'createLayerFunctionInterpolators','CylinderNeighbors','CapNeighbors','Neighbors'},
        'corrections.py': {'submask','HelixCorrector'},
    }
    for name, names in selections.items():
        path = source/'trackml_solution'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins['files']['trackml_solution/'+name]
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-'+name+'>','exec'),ns)
    return ns


def source_evaluator(ns, maps, cylinders=3, caps=4):
    # Only source interpolation methods need initialization; no hit data or kNN
    # state is used by these methods. All method bodies remain unmodified.
    evaluator = ns['Neighbors'].__new__(ns['Neighbors'])
    evaluator.layer_functions = maps
    groups = []
    for outer, inner, count, cylinder in [('CylinderNeighbors','Cylinder',cylinders,True),
                                         ('CapNeighbors','Cap',caps,False)]:
        cls = getattr(ns[outer],inner); layers = []
        for layer_id in range(count):
            layer = cls.__new__(cls)
            layer.layer_functions = ns['createLayerFunctionInterpolators'](cylinder,layer_id,maps)
            layers.append(layer)
        groups.append(layers)
    evaluator.cyln = SimpleNamespace(cylinders=groups[0])
    evaluator.capn = SimpleNamespace(caps=groups[1])
    return evaluator


def validate(root, source):
    ns = source_namespace(root,source)
    counts = dict(interpolation_cases=0, analytic_interpolation_arrays=0,
                  corrected_intersections=0, recursive_calls=0, changed_coordinates=0)
    def equal(a,b):
        assert len(a)==len(b)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    rng = np.random.RandomState(931)
    for dims,drop in [(1,0),(2,0),(3,0),(2,1),(3,1),(3,2)]:
        for missing in [False,True]:
            maps = SyntheticMaps(dims,drop,missing)
            a = build_layer_evaluator(maps,cylinder_count=3,cap_count=4)
            b = source_evaluator(ns,maps)
            points = rng.uniform(-250,250,(64,dims))  # includes extrapolation
            ids = np.arange(64)%3
            for cylinder in [True,False,rng.choice([False,True],64)]:
                actual = a.evaluateLayerFunctions(None,None,None,cylinder,ids,points=points)
                expected = b.evaluateLayerFunctions(None,None,None,cylinder,ids,points=points)
                equal(actual,expected); counts['interpolation_cases']+=1
                families = np.broadcast_to(cylinder,ids.shape)
                for j,fn in enumerate(maps.functions):
                    oracle = np.full(64,np.nan)
                    if fn != 'absent':
                        for family in [False,True]:
                            for layer in range(3):
                                select = (families==family)&(ids==layer)
                                if missing and layer==1: continue
                                oracle[select] = maps.formula(points[select],family,layer,fn)
                    np.testing.assert_allclose(actual[j],oracle,atol=1e-13,rtol=1e-13,equal_nan=True)
                    counts['analytic_interpolation_arrays']+=1
            # Source-derived coordinate mapping applies to one/two-dimensional
            # grids with no dropped dimensions; explicit points cover other grids.
            if dims <= 2 and drop == 0:
                xyz = rng.uniform(1,250,(3,64)); xyz[0,0]=np.nan
                families = rng.choice([False,True],64)
                equal(a.evaluateLayerFunctions(*xyz,families,ids),
                      b.evaluateLayerFunctions(*xyz,families,ids))
                counts['interpolation_cases']+=1
    assert createLayerFunctionInterpolators(True,0,None) is None
    # An interior singleton dimension is unsupported in the source, too.
    class InteriorSingleton(SyntheticMaps):
        def getGridValues(self, **kwargs):
            return [([np.array([0.,1.]),np.array([0.])],np.zeros((2,1)))]*3
    for create in [createLayerFunctionInterpolators,ns['createLayerFunctionInterpolators']]:
        try: create(True,0,InteriorSingleton())
        except AssertionError: pass
        else: raise AssertionError('Interior singleton must retain source rejection')

    modules = synthetic_modules(scale=10)
    with threadpool_limits(limits=1):
        actual_spec = discover_detector(modules)
        expected_spec = ns['DetectorSpec'](SimpleNamespace(detectors_df=modules))

    def traced(obj):
        calls=[]; original=obj.intersectHelices
        def call(*args,**kwargs):
            calls.append((len(args[0]),kwargs.get('iterations',10)))
            return original(*args,**kwargs)
        obj.intersectHelices=call
        return calls

    for seed in range(8):
        rng=np.random.RandomState(1031+seed);n=128
        xm=rng.uniform(-300,300,n);ym=rng.uniform(-300,300,n)
        radius=rng.uniform(10,500,n);phi=rng.uniform(-np.pi,np.pi,n)
        z=rng.uniform(-450,450,n);direction=rng.choice([-1.,1.],n)
        pitch=rng.uniform(20,2000,n)*rng.choice([-1.,1.],n)
        values=[xm+radius*np.cos(phi),ym+radius*np.sin(phi),z,direction,xm,ym,radius,pitch]
        before=[v.copy() for v in values]
        params=(direction,xm,ym,radius,pitch)
        previous=(xm+radius*np.cos(phi-.1),ym+radius*np.sin(phi-.1),z-pitch*.1/(2*np.pi))
        def corrector(cls,spec,evaluator,maps):
            run=SimpleNamespace(layer_functions=maps,neighbors=evaluator)
            return cls(SimpleNamespace(spec=spec),run,
                       hel_params=tuple(p.copy() for p in params),
                       hel_points=[tuple(p.copy() for p in previous),tuple(p.copy() for p in values[:3])])
        for missing in [False,True]:
            maps=SyntheticMaps(missing=missing)
            ae=build_layer_evaluator(maps,cylinder_count=3,cap_count=4)
            be=source_evaluator(ns,maps)
            for missable in [False,True]:
                for force in [None,rng.choice([False,True],n)]:
                    ac=corrector(HelixCorrector,actual_spec,ae,maps)
                    bc=corrector(ns['HelixCorrector'],expected_spec,be,maps)
                    ai=Intersector(actual_spec);bi=ns['Intersector'](expected_spec)
                    atrace=[traced(ai.cylis),traced(ai.capis)]
                    btrace=[traced(bi.cylis),traced(bi.capis)]
                    options=dict(missable=missable,force_cyl_closer=force)
                    with np.errstate(invalid='ignore',divide='ignore'):
                        a=ai.findNextHelixIntersection(*values,corrector=ac,**options)
                        b=bi.findNextHelixIntersection(*values,corrector=bc,**options)
                        plain=Intersector(actual_spec).findNextHelixIntersection(*values,**options)
                    equal(a,b);equal(ac.helixParams(),bc.helixParams())
                    assert atrace==btrace
                    counts['recursive_calls']+=sum(len(t)-1 for t in atrace)
                    counts['corrected_intersections']+=1
                    for x,y in zip(a[:3],plain[:3]):
                        counts['changed_coordinates']+=int(np.sum(np.isfinite(x)&np.isfinite(y)&(x!=y)))
        equal(values,before)
    assert counts['recursive_calls']>0 and counts['changed_coordinates']>0
    paths=['sciona/trackml_layer_functions.py','sciona/trackml_corrections.py',
           'sciona/trackml_intersections.py','sciona/trackml_detector.py',
           'sciona/trackml_geometry.py','sciona/trackml_helix.py',
           'scripts/validate_trackml_layer_integration.py','scripts/validate_trackml_detector.py',
           'docs/reviews/competition_trackml_source_pins.json','docs/licenses/TrackML-BSD-2-Clause.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Analytic synthetic grids only; no learned maps, detector data or accuracy claim.',
                     'Source correction/intersection methods do not call updateHelices; candidate lifecycle must validate its caller.',
                     'Source interpolation extrapolates outside grids; unsupported interior singleton dimensions still reject.',
                     'Neighbor search, Bayesian scoring, candidate lifecycle and CDG publication remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_trackml_layer_integration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
