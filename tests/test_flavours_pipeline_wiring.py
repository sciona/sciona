"""Reject incomplete physical or training contracts before fitting begins."""
import numpy as np
import pytest
from sciona.atoms.ml.flavours_pipeline_wiring import training_prediction_inputs


def inputs():
    def physical(rows):
        return [np.ones((rows,46)),np.ones((rows,3)),np.full((rows,3),.5),
                np.ones(rows),np.ones(rows),np.ones(rows)]
    return dict(training_inputs=physical(600),query_inputs=physical(7),
        labels=np.tile(np.array([0,1],dtype=np.int64),300),mass=np.ones(600),excluded_column=17)


@pytest.mark.parametrize('fault',['labels_dtype','labels_alignment','mass_alignment','missing_array',
    'daughter_count','negative_magnitude','base_width','nonfinite','insufficient_inner_classes','bad_exclusion'])
def test_preflight_rejects_invalid_complete_pipeline_inputs(fault):
    data=inputs()
    if fault=='labels_dtype':data['labels']=data['labels'].astype(float)
    elif fault=='labels_alignment':data['labels']=data['labels'][:-1]
    elif fault=='mass_alignment':data['mass']=data['mass'][:-1]
    elif fault=='missing_array':data['query_inputs']=data['query_inputs'][:-1]
    elif fault=='daughter_count':data['query_inputs'][1]=np.ones((7,4))
    elif fault=='negative_magnitude':data['query_inputs'][2][0,0]=-1
    elif fault=='base_width':data['training_inputs'][0]=np.ones((600,45))
    elif fault=='nonfinite':data['mass'][0]=np.nan
    elif fault=='insufficient_inner_classes':data['labels'][:250]=0
    else:data['excluded_column']=True
    with pytest.raises(ValueError):training_prediction_inputs(**data)


def test_preflight_returns_exact_ordered_inputs_and_training_targets_only():
    data=inputs();output=training_prediction_inputs(**data)
    assert len(output)==15
    for actual,expected in zip(output[:12],data['training_inputs']+data['query_inputs']):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(output[12],data['labels'])
    np.testing.assert_array_equal(output[13],data['mass'])
    assert output[14]==17
