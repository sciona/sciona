import numpy as np
import pytest
from sciona.atoms.ml.calibration.float32_correction import apply_float32_offsets
from sciona.atoms.ml.model_selection.training_contracts import training_inputs, prediction_inputs, append_feature_name


def test_float32_signed_offsets_and_threshold_tie():
    p=np.array([10.,10.,10.],dtype=np.float32)
    q=np.array([.9,.5,.1],dtype=np.float32)
    actual=apply_float32_offsets(p,q,np.array([-2.,3.]),.5)
    np.testing.assert_array_equal(actual,[8.,10.,7.])
    assert actual.dtype==np.float32
    np.testing.assert_array_equal(p,[10.,10.,10.])


def test_float32_offset_cast_and_result_overflow_rejected():
    p=np.array([1.],dtype=np.float32);q=np.array([.9],dtype=np.float32)
    with pytest.raises(ValueError): apply_float32_offsets(p,q,np.array([1e100,1.]),.5)
    p[0]=np.finfo(np.float32).max
    with pytest.raises(ValueError): apply_float32_offsets(p,q,np.array([float(p[0]),1.]),.5)
    with pytest.raises(ValueError): apply_float32_offsets(p,np.array([1.1],dtype=np.float32),np.array([1.,1.]),.5)


def test_training_contract_checks_common_alignment_and_prediction_schema():
    features=np.ones((4,2));targets=np.arange(4,dtype=np.float64);groups=np.arange(4,dtype=np.int64)
    result=training_inputs(features,targets,groups,['a','b'])
    assert len(result)==4 and result[3]==['a','b']
    assert prediction_inputs(np.ones((7,2)),['a','b']).shape==(7,2)
    with pytest.raises(ValueError): training_inputs(features,targets[:-1],groups,['a','b'])
    with pytest.raises(ValueError): prediction_inputs(np.ones((7,3)),['a','b'])


def test_classifier_schema_append_rejects_duplicate_and_preserves_order():
    names=['b','a']
    assert append_feature_name(names,'predicted')==['b','a','predicted'] and names==['b','a']
    with pytest.raises(ValueError): append_feature_name(names,'a')
