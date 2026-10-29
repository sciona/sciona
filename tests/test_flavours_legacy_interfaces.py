"""Legacy mirrors preserve optional native modes without claiming qualification."""
from scripts.reconcile_flavours_legacy_drafts import ports
from sciona.atoms.ml.sklearn.preprocessing.atoms import (
    standard_scaler_fit, standard_scaler_partial_fit, standard_scaler_transform,
)
from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
    compute_cvm_mass_decorrelation, roc_auc_truncated_weighted,
)


def inputs(function):
    return {p['name']:p for p in ports(function) if p['direction']=='input'}


def test_scaler_optional_weights_and_streaming_state_are_not_required():
    for function in (standard_scaler_fit,standard_scaler_partial_fit):
        declared=inputs(function)
        assert declared['X']['required']
        assert declared['sample_weight']['default_value_repr']=='None'
        assert not declared['sample_weight']['required']
        assert 'None' in declared['sample_weight']['type_desc']
        assert declared['with_mean']['default_value_repr']=='True'
        assert declared['with_std']['default_value_repr']=='True'
    state=inputs(standard_scaler_partial_fit)['state']
    assert not state['required'] and state['default_value_repr']=='None'
    assert state['type_desc']=='StandardScalerState | None'


def test_transform_requires_native_state_and_preserves_copy_default():
    declared=inputs(standard_scaler_transform)
    assert declared['state']['required']
    assert declared['state']['type_desc']=='StandardScalerState'
    assert declared['copy']['default_value_repr']=='True'
    assert ports(standard_scaler_transform)[-1]['type_desc']=='MatrixLike'


def test_metric_defaults_preserve_window_and_weighted_auc_semantics():
    cvm=inputs(compute_cvm_mass_decorrelation)
    assert cvm['n_neighbours']['default_value_repr']=='200'
    assert cvm['step']['default_value_repr']=='50'
    auc=inputs(roc_auc_truncated_weighted)
    assert auc['tpr_thresholds']['default_value_repr']=='(0.2, 0.4, 0.6, 0.8)'
    assert auc['weights']['default_value_repr']=='(4, 3, 2, 1, 0)'
    assert auc['weights']['type_desc']=='tuple[float, ...]'
