"""Synthetic complete-feature OOF training and held-out mutation checks."""
import numpy as np
import pytest

from sciona.tabular_complete_training import fit_complete_stacker
from tests.test_tabular_ensemble_training import sample

POLICY = dict(clip_low=0., clip_high=1., max_mode_fraction=1., log_columns=[0], interactions=[], smoothing=1., group_aggregates=[])


def test_aggregate_maps_exclude_outer_validation_values():
    n, c, y, f, g = sample()
    policy = dict(POLICY, group_aggregates=[dict(key_column=0, value_columns=[0],
        statistics=['mean', 'count'], unknown_value=-1.)])
    first = fit_complete_stacker(n, c, y, f, g, feature_controls=policy, trees=8)
    changed = [list(row) for row in n]
    for i, fold in enumerate(f):
        if fold == 0:
            changed[i][0] = 100000.
    second = fit_complete_stacker(changed, c, y, f, g, feature_controls=policy, trees=8)
    a = first.fold_models[0].features.aggregates[0][1]
    b = second.fold_models[0].features.aggregates[0][1]
    assert dict(a.mapping) == dict(b.mapping)


def fit(values=None):
    return fit_complete_stacker(*(sample() if values is None else values), feature_controls=POLICY, trees=8)


def test_complete_oof_and_full_refit_use_new_feature_state():
    n,c,y,f,g = sample()
    model = fit()
    assert model.oof_predictions.shape == (12, 2)
    assert model.base.features.base.rows == 12
    for fold, base in model.fold_models.items():
        held = [i for i, value in enumerate(f) if value == fold]
        assert base.features.base.rows == 8
        assert base.features.log_columns == (0,)
        assert base.features.categories is not None
        assert base.trees.n_jobs == 1
        np.testing.assert_array_equal(model.oof_predictions[held], base.predict([n[i] for i in held], [c[i] for i in held]))


def test_heldout_labels_numeric_values_and_categories_do_not_fit_fold_model():
    n,c,y,f,g = sample()
    baseline = fit()
    changed_n = [row[:] for row in n]; changed_c = [row[:] for row in c]; changed_y = y[:]
    for i in range(4):
        changed_n[i] = [999., -500.]
        changed_c[i] = ['heldout-only']
        changed_y[i] = 1-y[i]
    changed = fit((changed_n,changed_c,changed_y,f,g))
    a,b = baseline.fold_models[0],changed.fold_models[0]
    np.testing.assert_array_equal(a.features.base.medians,b.features.base.medians)
    assert tuple(dict(m) for m in a.features.categories.target_means) == tuple(dict(m) for m in b.features.categories.target_means)
    np.testing.assert_array_equal(a.linear[-1].coef_, b.linear[-1].coef_)
    np.testing.assert_array_equal(a.predict(n[:4], c[:4]), b.predict(n[:4], c[:4]))


def test_base_model_fit_receives_excluded_matrix_not_full_training_maps(monkeypatch):
    import sciona.tabular_complete_training as training
    real = training.fit_binary_tabular_features
    supplied = []
    def capture(*args, **kwargs):
        state, matrix = real(*args, **kwargs)
        supplied.append((state, matrix.copy(), state.transform(args[0], args[1])))
        return state, matrix
    monkeypatch.setattr(training, 'fit_binary_tabular_features', capture)
    model = fit()
    fitted = list(model.fold_models.values())+[model.base]
    assert len(supplied) == len(fitted) == 4
    assert any(not np.array_equal(matrix, full) for _,matrix,full in supplied)
    for base,(state,matrix,full) in zip(fitted,supplied,strict=True):
        assert base.features is state
        # StandardScaler records the exact mean of the matrix used for fitting.
        np.testing.assert_allclose(base.linear[0].mean_, matrix.mean(axis=0), rtol=0, atol=0)
        np.testing.assert_allclose(base.linear[0].var_, matrix.var(axis=0), rtol=1e-14, atol=1e-15)


def test_complete_training_is_repeatable():
    a,b = fit(),fit()
    np.testing.assert_array_equal(a.oof_predictions,b.oof_predictions)
    np.testing.assert_array_equal(a.predict([[4.,None]],[['new']]), b.predict([[4.,None]],[['new']]))


def test_complete_features_support_disjoint_calibration_and_frozen_query():
    from sciona.tabular_ensemble_calibration import calibrate_stack, predict_calibrated
    model = fit()
    before = tuple(dict(m) for m in model.base.features.categories.target_means)
    calibrated = calibrate_stack(model, [[0.,0.],[1.,0.],[4.,1.],[7.,3.]],
        [['category0'],['category1'],['new'],['category2']], [0,1,0,1], ['cal0','cal1','cal2','cal3'])
    result = predict_calibrated(calibrated, [[2.,0.],[7.,1.]], [['category0'],['new']], ['query0','query1'])
    assert len(result['probabilities']) == len(result['classes']) == 2
    assert all(0 <= p <= 1 for p in result['probabilities'])
    assert result['classes'] == [int(p >= .5) for p in result['probabilities']]
    assert before == tuple(dict(m) for m in model.base.features.categories.target_means)
    with pytest.raises(ValueError, match='overlap'):
        predict_calibrated(calibrated, [[2.,0.]], [['new']], ['cal0'])


@pytest.mark.parametrize('case', ['crossing_group','single_fold','missing_policy'])
def test_invalid_splits_or_policy(case):
    n,c,y,f,g = sample(); policy = dict(POLICY)
    if case == 'crossing_group': g[5] = g[0]
    if case == 'single_fold': f = [0]*12
    if case == 'missing_policy': del policy['smoothing']
    with pytest.raises(ValueError):
        fit_complete_stacker(n,c,y,f,g,feature_controls=policy,trees=8)
