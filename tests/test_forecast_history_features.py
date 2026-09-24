import numpy as np
import pytest
from sciona.forecast_history_features import forecast_history_features as compute


def args():
    return [np.array([0,0,10,10,20,20],dtype=np.int64),
        np.array([10,20,10,20,20,30],dtype=np.int64),
        np.array([[1],[3],[2],[6],[9],[12]],dtype=np.float64),
        np.array([-1,10,20],dtype=np.int64)]


def policy():
    return dict(history_window=20,lead_bands=[(None,10)],
        contrast_leads=np.array([[0,10]],dtype=np.int64),revision_lags=np.array([1,2],dtype=np.int64))


def test_independent_history_bands_contrasts_and_revisions():
    r=compute(*args(),**policy())
    np.testing.assert_array_equal(r['issue_available'],[False,True,True])
    np.testing.assert_allclose(r['mean'][:,0],[np.nan,1.5,5.5],equal_nan=True)
    np.testing.assert_allclose(r['min'][:,0],[np.nan,1,2],equal_nan=True)
    np.testing.assert_allclose(r['max'][:,0],[np.nan,2,9],equal_nan=True)
    np.testing.assert_allclose(r['bands'][:,0,0],[np.nan,4,10.5],equal_nan=True)
    np.testing.assert_allclose(r['contrasts'][:,0,0],[np.nan,-4,-3],equal_nan=True)
    np.testing.assert_allclose(r['latest_valid'][:,0],[np.nan,2,9],equal_nan=True)
    np.testing.assert_allclose(r['revisions'][:,:,0],[[np.nan,np.nan],[1,np.nan],[3,6]],equal_nan=True)


def test_future_issue_isolation_order_and_duplicate_queries():
    values=args();expected=compute(*values,**policy())
    values[2][4:]+=1000
    actual=compute(*values,**policy())
    for key in expected:np.testing.assert_array_equal(actual[key][:2],expected[key][:2])
    order=[5,2,0,4,1,3]
    shuffled=compute(values[0][order],values[1][order],values[2][order],values[3][[1,0,1]],**policy())
    for key in expected:np.testing.assert_array_equal(shuffled[key],expected[key][[1,0,1]])


def test_missing_horizon_never_borrows_an_older_issue():
    values=args();values[:3]=[value[:5] for value in values[:3]]
    result=compute(*values,**policy())
    assert np.isnan(result['contrasts'][2]).all()


def test_demand_forecast_reuse_under_clock_rescaling():
    values=args();options=policy();expected=compute(*values,**options)
    for slot in [0,1,3]:values[slot]*=60
    options['history_window']*=60;options['lead_bands']=[(None,600)];options['contrast_leads']*=60
    actual=compute(*values,**options)
    for key in expected:np.testing.assert_allclose(actual[key],expected[key],equal_nan=True)


def test_duplicate_issue_valid_pair_rejected():
    values=args();values[1][1]=values[1][0]
    with pytest.raises(ValueError,match='Unique issue'):compute(*values,**policy())


def test_missing_field_values_and_empty_history():
    values=args();values[2][:]=np.nan
    result=compute(*values,**policy())
    assert all(np.isnan(v).all() for k,v in result.items() if k!='issue_available')
    empty=compute(values[0][:0],values[1][:0],values[2][:0],values[3],**policy())
    assert not empty['issue_available'].any()
