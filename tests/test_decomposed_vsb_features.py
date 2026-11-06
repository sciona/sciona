"""Synthetic source parity and independent edge checks for numeric decomposition."""
import numpy as np
import pytest
from sciona.atoms.scipy.signal import lfilter
from sciona.vsb_baseline import subtract_baseline
from sciona.vsb_peaks import window_maxima,select_peaks
from sciona.vsb_descriptors import peak_descriptors,process_signal
from sciona.decomposed_vsb_features import (baseline_arguments,baseline_residual,rank_peaks,
    amplitude_gradient,select_ranked_peaks)


def decomposed(values,window=25):
    b,a,x,axis,zi=baseline_arguments(values,.99)
    residual=baseline_residual(lfilter(b,a,x,axis=axis,zi=zi))
    positions,heights=rank_peaks(residual,window_maxima(np.abs(residual),window=window))
    gradient=amplitude_gradient(heights,9,8)
    selected,amplitudes,knee=select_ranked_peaks(positions,heights,gradient,-.01,1000,4)
    return dict(positions=selected,heights=amplitudes,residual=residual,knee=knee,candidate_count=len(positions))


@pytest.mark.parametrize('seed',[11,87,91])
@pytest.mark.parametrize('size',[512,4096])
@pytest.mark.parametrize('window',[7,25])
def test_complete_peak_and_descriptor_parity(seed,size,window):
    values=np.random.default_rng(seed).normal(size=size);saved=values.copy()
    expected=select_peaks(values,window=window);actual=decomposed(values,window)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
    if window==7:
        # These fixtures deliberately violate the retained descriptor alignment
        # boundary when the detection window is narrower than its support.
        with pytest.raises(ValueError,match='maximum must align'):
            process_signal(values,window=window)
        with pytest.raises(ValueError,match='maximum must align'):
            peak_descriptors(actual['positions'],actual['residual'])
    else:
        expected_features=process_signal(values,window=window)
        np.testing.assert_array_equal(peak_descriptors(actual['positions'],actual['residual']),expected_features['descriptors'])
    np.testing.assert_array_equal(values,saved)


@pytest.mark.parametrize('retention',[0.,.5,.99,1.])
def test_filter_boundary_general_retention(retention):
    values=np.array([3,1,-4,9,2,2],dtype=np.int64)
    b,a,x,axis,zi=baseline_arguments(values,retention)
    filtered=lfilter(b,a,x,axis=axis,zi=zi);saved=filtered.copy()
    result=baseline_residual(filtered)
    np.testing.assert_array_equal(result,subtract_baseline(values,retention=retention))
    np.testing.assert_array_equal(filtered,saved)
    assert not np.shares_memory(result,filtered) and not np.shares_memory(values,x)


def test_explicit_gradient_and_negative_slice():
    h=np.array([9.,7.,6.,2.,1.]);p=np.array([10,8,6,4,2])
    np.testing.assert_array_equal(amplitude_gradient(h,2,0),[-1.,-1.5,-2.5,-2.5,-.5])
    np.testing.assert_array_equal(amplitude_gradient(h,2,1),[-1.5,-2.5,-2.5])
    positions,heights,knee=select_ranked_peaks(p,h,[],0.,3,2)
    assert knee==-2
    np.testing.assert_array_equal(positions,[6,8,10]);np.testing.assert_array_equal(heights,[6.,7.,9.])
    assert amplitude_gradient(h,2,99).size==0


@pytest.mark.parametrize('bad',[True,-.1,1.1,np.nan])
def test_invalid_retention(bad):
    with pytest.raises(ValueError):baseline_arguments([1.,2.],bad)


@pytest.mark.parametrize('positions',[[1],[1,1],[-1,1],[1,3],[True,False],[0.,1.]])
def test_invalid_candidates(positions):
    with pytest.raises(ValueError):rank_peaks([1.,2.,3.],positions)


@pytest.mark.parametrize('heights,width,trim',[([1.,2.],2,0),([1.],2,0),([2.,-1.],2,0),([2.,1.],0,0),([2.,1.],True,0),([2.,1.],2,-1)])
def test_invalid_gradient_controls(heights,width,trim):
    with pytest.raises(ValueError):amplitude_gradient(heights,width,trim)


def test_nonfinite_filter_result_rejected():
    with pytest.raises(ValueError):baseline_residual([0.,np.inf])


def test_complete_phase_and_measurement_feature_parity():
    from sciona.vsb_measurement import fundamental,phase_quadrants,aggregate,measurement_features
    values=np.random.default_rng(101).normal(size=(512,9));saved=values.copy()
    coefficients=fundamental(values);groups=[];heights=[];descriptors=[];quadrants=[]
    for column in range(values.shape[1]):
        result=decomposed(values[:,column])
        groups.extend([column//3]*len(result['positions']))
        heights.extend(result['heights'])
        descriptors.extend(peak_descriptors(result['positions'],result['residual']))
        quadrants.extend(phase_quadrants(result['positions'],coefficients[column],length=len(values)))
    actual=aggregate(np.asarray(groups,dtype=int),np.asarray(heights),
        np.asarray(descriptors).reshape(-1,4),np.asarray(quadrants,dtype=int),measurement_count=3)
    np.testing.assert_array_equal(actual,measurement_features(values))
    np.testing.assert_array_equal(values,saved)
