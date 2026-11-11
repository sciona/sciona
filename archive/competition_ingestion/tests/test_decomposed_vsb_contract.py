"""Synthetic complete runtime parity with explicit numeric and fold bindings."""
import copy
from unittest.mock import patch
import numpy as np
import pytest
from sciona.atoms.scipy.signal import lfilter
from sciona import decomposed_vsb_features as features
from sciona import decomposed_vsb_training as training
from sciona.decomposed_vsb_contract import unpack,encode_result
from sciona.decomposed_vsb_measurement import collect
from sciona.vsb_peaks import window_maxima
from sciona.vsb_descriptors import peak_descriptors
from sciona.vsb_measurement import fundamental,phase_quadrants,aggregate
from sciona.vsb_threshold import repeated_folds,threshold
from sciona.vsb_execution import prepare,execute
from scripts.vsb_synthetic import payload

FUNCTIONS=(features.baseline_arguments,lfilter,features.baseline_residual,window_maxima,
    features.rank_peaks,features.amplitude_gradient,features.select_ranked_peaks,peak_descriptors,phase_quadrants)


def test_complete_raw_runtime_and_all_injected_operations():
    raw=payload();saved=copy.deepcopy(raw)
    signals,labels,query,controls,fit_controls=unpack(raw)
    counts=[0]*len(FUNCTIONS)
    def track(i,fn):
        def invoke(*args,**kwargs):counts[i]+=1;return fn(*args,**kwargs)
        return invoke
    functions=[track(i,fn) for i,fn in enumerate(FUNCTIONS)]
    matrices=[]
    for matrix in (signals,query):
        groups,heights,descriptors,quadrants,size=collect(matrix,fundamental(matrix),controls,*functions)
        matrices.append(aggregate(groups,heights,descriptors,quadrants,measurement_count=size))
    assert counts==[signals.shape[1]+query.shape[1]]*len(FUNCTIONS)
    x,y,q,signal_labels=training.recipe_inputs(matrices[0],labels,matrices[1])
    plans=repeated_folds(y,seed=fit_controls['split_seed'],repetitions=fit_controls['repetitions']);calls=[]
    def fold(*args):calls.append(1);return training.fit_fold(*args)
    trained=training.ensemble(x,y,q,plans,fit_controls['repetitions'],fit_controls['tree_seed'],
        fit_controls['rounds'],fit_controls['patience'],fold)
    selected=threshold(*training.threshold_inputs(signal_labels,trained['training_probabilities']))
    actual=encode_result(training.result(trained,selected))
    assert len(calls)==125 and actual==execute(prepare(raw)) and raw==saved


def test_unpack_copies_without_hidden_feature_or_training_execution():
    raw=payload()
    with (patch('sciona.vsb_measurement.measurement_features',side_effect=AssertionError('Hidden feature extraction')),
          patch('sciona.vsb_training.fit',side_effect=AssertionError('Hidden training'))):
        signals,labels,query,_,_=unpack(raw)
    signals[0,0]=99;labels[0,0]=1;query[0,0]=99
    assert raw['training']['signals'][0][0]!=99 and raw['training']['labels'][0][0]==0
    assert raw['query']['signals'][0][0]!=99


@pytest.mark.parametrize('fault',['coefficients','triples','controls','callable'])
def test_bad_collection_boundary(fault):
    x=np.ones((32,3));coefficients=fundamental(x)
    controls=dict(retention=.99,window=25,width=9,trim=8,threshold=-.01,count=1000,offset=4,small_radius=5,large_radius=25)
    functions=list(FUNCTIONS)
    if fault=='coefficients':coefficients=coefficients[:1]
    elif fault=='triples':x=x[:,:2]
    elif fault=='controls':controls.pop('offset')
    else:functions[0]=None
    with pytest.raises(ValueError):collect(x,coefficients,controls,*functions)
