"""Reusable regression validation and native feature-importance operations."""
import numpy as np
import pandas as pd


def mean_absolute_error(observed, predicted):
    left, right = np.asarray(observed,dtype=np.float64), np.asarray(predicted,dtype=np.float64)
    if left.ndim != 1 or not left.size or left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Nonempty finite aligned scalar targets and predictions required')
    with np.errstate(over='raise',invalid='raise'):
        result = float(np.abs(left-right).mean())
    if not np.isfinite(result):raise ValueError('Nonfinite validation error')
    return result


def accept_model_below(model, score, ceiling):
    if model is None or not np.isfinite(score) or not np.isfinite(ceiling) or not score < ceiling:
        raise ValueError('Model score must be finite and strictly below the acceptance ceiling')
    return model


def native_feature_importance(model):
    names = list(model.feature_names_)
    values = np.asarray(model.get_feature_importance(thread_count=1),dtype=np.float64)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError('Finite native importance aligned to feature names required')
    return names,values.copy()


def normalize_ranked_importance(names, values):
    values = np.asarray(values,dtype=np.float64)
    if not isinstance(names,list) or not names or len(set(names)) != len(names) or any(not isinstance(name,str) or not name for name in names):
        raise ValueError('Unique nonempty feature names required')
    if values.shape != (len(names),) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Aligned finite nonnegative importance required')
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Positive finite total importance required')
    return pd.DataFrame({'feature':names,'importance':values/total}).sort_values('importance',ascending=False)
