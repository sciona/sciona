import numpy as np
import pytest
from sciona.prediction_fallback_chain import prediction_fallback_chain as compute


def fail():raise RuntimeError('Synthetic predictor failure')


def test_primary_success_is_owned_and_skips_baseline():
    original=np.array([1,2])
    result,route=compute(lambda:original,fail,2,30)
    assert route=='primary' and not np.shares_memory(result,original)
    np.testing.assert_array_equal(result,original)


def test_baseline_preserves_fractional_results():
    result,route=compute(fail,lambda:np.array([1.,45.5]),2,30)
    assert route=='baseline'
    np.testing.assert_array_equal(result,[1.,45.5])


@pytest.mark.parametrize('bad',[np.array([np.nan]),np.array([1,2]),np.array(['value']),None])
def test_invalid_primary_and_failed_baseline_select_constant(bad):
    result,route=compute(lambda:bad,fail,1,30)
    assert route=='constant'
    np.testing.assert_array_equal(result,[30.])


@pytest.mark.parametrize('exception',[KeyboardInterrupt,SystemExit])
def test_process_control_exceptions_propagate(exception):
    def interrupt():raise exception()
    with pytest.raises(exception):compute(interrupt,fail,1,30)
    with pytest.raises(exception):compute(fail,interrupt,1,30)


def test_empty_queries_skip_callbacks():
    result,route=compute(fail,fail,0,30)
    assert route=='empty' and result.shape==(0,)
