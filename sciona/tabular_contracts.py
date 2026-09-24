"""Stable feature ordering shared by training and prediction adapters."""
import numpy as np
import pandas as pd


def stable_feature_order(names):
    """Deduplicate feature names in first-occurrence order, without hash ordering.

    Persist this ordered list with the model and reuse it at prediction time.
    It identifies matrix columns; it does not assert their physical units or
    interpretation. Those remain part of the caller's feature contract.
    """
    if not isinstance(names, (list, tuple)) or not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('Nonempty sequence of nonempty feature names required')
    return list(dict.fromkeys(names))


def ordered_feature_matrix(frame, feature_order):
    """Select finite numeric features in an explicit, unique persisted order.

    Extra columns are ignored; missing or duplicated columns are errors. Returns
    an independent float64 matrix. There is no imputation or scaling, and no
    inference of order from the incoming frame. Empty row populations are valid.
    """
    names = stable_feature_order(feature_order)
    if len(names) != len(feature_order):
        raise ValueError('Persisted feature order must be unique')
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not set(names) <= set(frame.columns):
        raise ValueError('Unique frame columns and all persisted features required')
    selected = frame[names]
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype) for dtype in selected.dtypes):
        raise ValueError('Real numeric features required')
    values = selected.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError('Finite features required; no implicit imputation')
    return values
