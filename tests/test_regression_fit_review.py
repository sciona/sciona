import numpy as np
import pytest
from sciona.regression_fit_review import mean_absolute_error,accept_model_below,normalize_ranked_importance


def test_error_and_strict_candidate_acceptance():
    assert mean_absolute_error([1.,4.],[3.,3.])==1.5
    model=object()
    assert accept_model_below(model,1.5,2.) is model
    with pytest.raises(ValueError):accept_model_below(model,2.,2.)
    with pytest.raises(ValueError):mean_absolute_error([1.],[1.,2.])


def test_ranked_normalization_retains_source_index():
    result=normalize_ranked_importance(['load','pressure','temperature'],np.array([1.,4.,5.]))
    assert result.index.tolist()==[2,1,0]
    assert result.feature.tolist()==['temperature','pressure','load']
    np.testing.assert_array_equal(result.importance,[.5,.4,.1])


@pytest.mark.parametrize('values',[[0.,0.],[-1.,2.],[np.nan,1.],[np.inf,1.]])
def test_undefined_importance_rejected(values):
    with pytest.raises(ValueError):normalize_ranked_importance(['a','b'],values)
