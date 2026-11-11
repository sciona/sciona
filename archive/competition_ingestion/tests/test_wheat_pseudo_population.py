import numpy as np
import pandas as pd
import pytest

from sciona.wheat_pseudo_population import build_pseudo_population


def inputs():
    original = pd.DataFrame(dict(image_id=['synthetic_a', 'synthetic_b', 'synthetic_c'],
        fold=[0, 1, 1], isbox=[True, True, False], xmin=[1., 2., np.nan], ymin=[1., 2., np.nan],
        xmax=[4., 6., np.nan], ymax=[5., 7., np.nan]), index=[10, 20, 30])
    predictions = {'synthetic_q': (np.empty((0, 4)), np.empty(0))}
    return original, predictions


def test_empty_query_is_training_negative_and_empty_heldout_row_is_excluded():
    original, predictions = inputs()
    before = original.copy(deep=True)
    global_state = np.random.get_state()
    train, valid = build_pseudo_population(original, ['synthetic_q', 'synthetic_q'], predictions,
        train_directory='synthetic_train', query_directory='synthetic_query', seed=8)
    assert len(train) == 2 and len(valid) == 1
    negative = train.loc[~train['isbox']]
    assert len(negative) == 1 and negative[['xmin', 'ymin', 'xmax', 'ymax']].isna().all().all()
    assert valid['image_path'].tolist() == ['synthetic_train/synthetic_b.jpg']
    pd.testing.assert_frame_equal(original, before)
    assert np.array_equal(global_state[1], np.random.get_state()[1])
    assert global_state[2:] == np.random.get_state()[2:]


@pytest.mark.parametrize('problem', ['missing_prediction', 'invalid_box', 'duplicate_index'])
def test_inconsistent_population_is_rejected(problem):
    original, predictions = inputs()
    if problem == 'missing_prediction':
        predictions.clear()
    elif problem == 'invalid_box':
        predictions['synthetic_q'] = (np.array([[5., 1., 2., 4.]]), np.array([.8]))
    else:
        original.index = [10, 10, 30]
    with pytest.raises(ValueError):
        build_pseudo_population(original, ['synthetic_q'], predictions,
            train_directory='synthetic_train', query_directory='synthetic_query', seed=8)
