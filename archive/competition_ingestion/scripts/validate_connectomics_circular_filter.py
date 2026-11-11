"""Qualify reusable filtering against pinned source and both corrected sweeps."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from sciona.atoms.numerical.spatial.circular_filter import ordered_circular_filter
from sciona.connectomics_source import source_namespace

ROOT=Path(__file__).resolve().parents[1]
TAPS={
    'f1':([0,-1,1],[1,1,1]),
    'f2':([0,1,2,3],[1,1,.8,.4]),
    'f3':([0,-1,-2,1],[1,1,1,1]),
    'f4':([0,-1,-2,-3],[1,1,1,1]),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(source):
    reference=source_namespace(source);reference['print']=lambda *a,**k:None
    rng=np.random.default_rng(1729);cases=0
    for dtype in (np.float32,np.float64):
        for shape in ((1,3),(8,3),(19,4),(7,2,3)):
            for layout in ('C','F'):
                values=np.array(rng.normal(size=shape),dtype=dtype,order=layout)
                before=values.copy()
                for name,(shifts,weights) in TAPS.items():
                    result=ordered_circular_filter(values,shifts,weights,0)
                    np.testing.assert_array_equal(result,reference[name](values))
                    np.testing.assert_array_equal(values,before)
                    assert result.dtype==values.dtype
                    cases+=1
    candidate=source_namespace(source);candidate['print']=lambda *a,**k:None
    calls={name:0 for name in TAPS}
    def filtered(name):
        def apply(values):
            calls[name]+=1
            return ordered_circular_filter(values,*TAPS[name],axis=0)
        return apply
    for name in TAPS:candidate[name]=filtered(name)
    values=np.asfortranarray(np.random.default_rng(1729).uniform(0,.8,(240,10)),dtype=np.float32)
    before=values.copy();sweeps=[]
    for mode in ('simple','tuned'):
        prior=dict(calls)
        expected=reference['make_'+mode+'_inference'](values)
        actual=candidate['make_'+mode+'_inference'](values)
        np.testing.assert_array_equal(actual,expected)
        directed=reference['make_prediction_directivity'](values,n_jobs=1)
        actual_directed=candidate['make_prediction_directivity'](values,n_jobs=1)
        np.testing.assert_array_equal(actual_directed,directed)
        np.testing.assert_array_equal(.997*actual+.003*actual_directed,.997*expected+.003*directed)
        used={name:calls[name]-prior[name] for name in calls}
        if used!={name:120 if mode=='tuned' or name in ('f1','f2') else 0 for name in TAPS}:
            raise ValueError('Complete 120-entry filter sweeps required')
        sweeps.append(dict(mode=mode,exact_scores=True,exact_optional_directivity_blend=True,
                           filter_calls=used,pca_fits=240 if mode=='simple' else 480))
    np.testing.assert_array_equal(values,before)
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        component_cases=cases,sweeps=sweeps,numpy_version=np.__version__,caller_inputs_preserved=True,
        provider_sha256=sha(inspect.getsourcefile(ordered_circular_filter)),
        validator_sha256=sha(__file__),test_sha256=sha(ROOT/'tests/test_ordered_circular_filter.py'),
        source_loader_sha256=sha(ROOT/'sciona/connectomics_source.py'),
        source_pins_sha256=sha(ROOT/'docs/reviews/competition_connectomics_source_pins.json'),
        reuse_contract='Explicit axis, ordered shifts and weights; no source filter names or domain semantics in the provider.',
        limitations=['Sweep integration uses replacement functions in the pinned corrected source namespace; a decomposed stored CDG is not yet built.',
                    'Source f1-f4 have exact synthetic agreement. Broader axis, signed-weight and validation behavior has separate synthetic unit tests.',
                    'Historical PCA and all other source recipe operations remain unchanged; this is not original-CDG publication approval.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    report=validate(parser.parse_args().source)
    (ROOT/'docs/reviews/connectomics_generic_circular_filter_source_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','component_cases','sweeps')}))
