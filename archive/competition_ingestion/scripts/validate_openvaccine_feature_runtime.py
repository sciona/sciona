"""Synthetic complete batch feature integration and folded-input failures."""
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
from sciona.openvaccine_features import folded_features
from sciona.openvaccine_structure import loop_labels
from validate_openvaccine_preprocessing import reference

ROOT=Path(__file__).resolve().parents[1]


def main():
    source='/private/tmp/sciona_openvaccine_source'
    sequences=['ACAAAAGU','AUAUAUAU','AGCUAGCU'];structures=['((....))','()()()()','........']
    rng=np.random.default_rng(654);probabilities=[]
    for _ in sequences:
        p=rng.uniform(0,.01,(8,8));p=p+p.T;np.fill_diagonal(p,0);probabilities.append(p)
    originals=[p.copy() for p in probabilities]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        actual=folded_features(source,sequences,structures,probabilities)
    expected=[reference(s,loop_labels(t),t,p) for s,t,p in zip(sequences,structures,probabilities)]
    for axis in range(2):np.testing.assert_allclose(actual[axis],np.concatenate([e[axis] for e in expected]),rtol=2e-7,atol=1e-8)
    for a,b in zip(probabilities,originals):np.testing.assert_array_equal(a,b)
    cases=[(['A'],['.'],[np.zeros((1,1))]),(['AX'],['..'],[np.zeros((2,2))]),
           (['AA'],['()'],[np.zeros((2,2))]),(['AU'],['(.'],[np.zeros((2,2))]),
           (['AU'],['..'],[np.ones((2,2))]),(['AU'],['..'],[np.array([[0,.1],[.2,0]])]),
           (['AU'],['..'],[np.array([[0,np.nan],[np.nan,0]])]),
           (['AU','ACGU'],['..','....'],[np.zeros((2,2)),np.zeros((4,4))])]
    for args in cases:
        try:folded_features(source,*args)
        except ValueError:pass
        else:raise AssertionError('Invalid folded input accepted')
    result=dict(status='passed',synthetic_only=True,batch_records=3,node_channels=55,adjacency_channels=8,
                independent_reference_comparison=True,probability_inputs_unchanged=True,invalid_cases=len(cases),
                scope='Complete loop-label and source feature batching given synthetic folded inputs; no folding engine execution.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_features.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_feature_runtime.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
