import numpy as np
import pandas as pd
import pytest
from sciona.materialized_prediction import predict_named_frame, average_prediction_vectors


class Predictor:
    feature_names_ = ['temperature', 'load']
    def predict(self, frame, *, thread_count):
        assert thread_count == 1
        return frame['temperature'].to_numpy()+frame['load'].to_numpy()


def test_native_schema_and_row_order():
    frame = pd.DataFrame({'temperature': [2., 8., 2.], 'load': [1., 3., 1.]}, index=[5, 1, 5])
    np.testing.assert_array_equal(predict_named_frame(Predictor(), frame), [3., 11., 3.])
    with pytest.raises(ValueError, match='schema'):
        predict_named_frame(Predictor(), frame[['load', 'temperature']])


def test_ordered_mean_and_nonmutation():
    value = np.array([1., 5.])
    result = average_prediction_vectors([value, np.array([3., 7.])])
    np.testing.assert_array_equal(result, [2., 6.])
    np.testing.assert_array_equal(value, [1., 5.])


@pytest.mark.parametrize('vectors', [[], [np.zeros(2), np.zeros(1)], [np.zeros((2, 1))], [np.array([np.nan])]])
def test_invalid_ensembles(vectors):
    with pytest.raises(ValueError):
        average_prediction_vectors(vectors)


def test_native_invalid_output():
    class Bad(Predictor):
        def predict(self, frame, *, thread_count):
            return np.zeros((len(frame), 1))
    with pytest.raises(ValueError, match='row-aligned'):
        predict_named_frame(Bad(), pd.DataFrame({'temperature': [2.], 'load': [1.]}))
