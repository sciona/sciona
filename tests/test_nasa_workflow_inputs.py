import numpy as np
import pandas as pd
import pytest
from scripts.validate_residual_classifier_graph import source_level_tables
from sciona.nasa_workflow_inputs import prepare_training_inputs,prepare_prediction_inputs,attach_predictions


@pytest.fixture(scope='module')
def records():
    rng=np.random.default_rng(9021)
    _,vocabulary,training=source_level_tables(rng,'KZZZ',True)
    _,_,prediction=source_level_tables(rng,'KZZZ',False,vocabulary)
    prepared,vocabulary,identities=prepare_training_inputs(training,'KZZZ')
    return training,prediction,prepared,vocabulary,identities


def test_training_and_prediction_preserve_schema_and_identity(records):
    training,prediction,prepared,vocabulary,identities=records
    assert len(identities)==480 and prepared['features'].shape[0]==480
    features,query_ids=prepare_prediction_inputs(prediction,'KZZZ',prepared['feature_names'],vocabulary)
    assert features.shape==(64,len(prepared['feature_names']))
    pd.testing.assert_frame_equal(query_ids,prediction['queries'][list(query_ids)])
    values=np.arange(64,dtype=np.int32)
    result=attach_predictions(query_ids,values)
    np.testing.assert_array_equal(result.minutes_until_pushback,values)
    assert 'minutes_until_pushback' not in query_ids


def test_missing_prediction_features_keep_rows_without_imputation(records):
    _,prediction,prepared,vocabulary,_=records
    missing={key:frame.copy() for key,frame in prediction.items()}
    identity=missing['queries'].gufi.iloc[0]
    missing['estimates']=missing['estimates'][missing['estimates'].gufi!=identity]
    missing['arrivals']=missing['arrivals'].iloc[:0]
    values,ids=prepare_prediction_inputs(missing,'KZZZ',prepared['feature_names'],vocabulary)
    assert len(ids)==64
    assert np.isnan(values[0,prepared['feature_names'].index('mean_departure_from_timepoint')])
    assert np.isnan(values[:,prepared['feature_names'].index('taxitime_to_gate_mean')]).all()


def test_query_order_restored_after_joins(records):
    _,prediction,prepared,vocabulary,_=records
    shuffled=dict(prediction)
    shuffled['queries']=prediction['queries'].iloc[::-1].reset_index(drop=True)
    _,ids=prepare_prediction_inputs(shuffled,'KZZZ',prepared['feature_names'],vocabulary)
    assert list(ids.gufi)==list(shuffled['queries'].gufi)


def test_wrong_slot_duplicate_queries_empty_training_and_output_mismatch_fail(records):
    training,prediction,prepared,vocabulary,_=records
    with pytest.raises(ValueError):prepare_training_inputs(training,'KYYY')
    duplicated=dict(prediction);duplicated['queries']=pd.concat([prediction['queries'],prediction['queries'].iloc[:1]])
    with pytest.raises(ValueError):prepare_prediction_inputs(duplicated,'KZZZ',prepared['feature_names'],vocabulary)
    empty=dict(training);empty['arrivals']=training['arrivals'].iloc[:0]
    with pytest.raises(ValueError,match='No complete'):prepare_training_inputs(empty,'KZZZ')
    with pytest.raises(ValueError):attach_predictions(prediction['queries'],np.ones(3,dtype=np.int32))
