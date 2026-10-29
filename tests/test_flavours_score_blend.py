"""Exact blend equation, role ordering and invalid-probability contracts."""
import numpy as np
import pytest
from sciona.flavours_blend import blend
from sciona.atoms.ml.flavours_score_blend import six_model_power_blend,witness_six_model_power_blend
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_exact_source_equation_including_extreme_probabilities(dtype):
    rng=np.random.default_rng(921);arrays=[rng.uniform(size=37).astype(dtype) for _ in range(6)]
    for a in arrays:a[:3]=[0,1,np.nextafter(dtype(1),dtype(0))]
    original=[a.copy() for a in arrays]
    expected=blend({f'xgb{i+1}':arrays[i] for i in range(5)},arrays[5])
    actual=six_model_power_blend(*arrays)
    np.testing.assert_array_equal(actual,expected)
    assert actual.dtype==np.float64 and np.ptp(actual)>0
    for a,b in zip(arrays,original):np.testing.assert_array_equal(a,b)
    switched=arrays.copy();switched[0],switched[1]=switched[1],switched[0]
    assert not np.array_equal(actual,six_model_power_blend(*switched))


@pytest.mark.parametrize('fault',['nan','negative','above_one','shape','integer'])
def test_invalid_probabilities_rejected(fault):
    values=[np.full(4,.5) for _ in range(6)]
    if fault=='nan':values[3][0]=np.nan
    elif fault=='negative':values[3][0]=-.1
    elif fault=='above_one':values[3][0]=1.1
    elif fault=='shape':values[3]=values[3][:-1]
    else:values[3]=values[3].astype(int)
    with pytest.raises(ValueError):six_model_power_blend(*values)


def test_witness_requires_dimensionless_aligned_inputs():
    values=[AbstractArray(shape=(4,),dtype='float32') for _ in range(6)]
    output=witness_six_model_power_blend(*values)
    assert output.dtype=='float64' and output.dim.is_dimensionless
    values[3]=AbstractArray(shape=(4,),dim=DimensionalSignature(M=1))
    with pytest.raises(ValueError,match='dimensionless'):witness_six_model_power_blend(*values)
