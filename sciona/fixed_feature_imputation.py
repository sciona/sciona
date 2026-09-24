"""Explicit fixed imputation and checked integer feature conversion."""
import numpy as np


def fixed_integer_features(values, fill_values, dtype='int16'):
    """Impute NaNs per column, truncate toward zero and reject integer overflow.

    `values` is a real numeric row-by-feature matrix. `fill_values` is an
    explicit finite vector in matching feature units, supplied from an approved
    policy or training-only state. This function never learns from query rows.
    It does not mutate either input. Only signed integer output types are allowed.
    """
    if not isinstance(values,np.ndarray) or values.ndim!=2 or values.dtype.kind not in 'iuf':
        raise ValueError('Real numeric feature matrix required')
    if not isinstance(fill_values,np.ndarray) or fill_values.ndim!=1 or fill_values.dtype.kind not in 'iuf' or len(fill_values)!=values.shape[1]:
        raise ValueError('Aligned per-feature fill vector required')
    target=np.dtype(dtype)
    if target.kind!='i':raise ValueError('Signed integer dtype required')
    for array in (values,fill_values):
        if array.dtype.kind in 'iu' and array.size and (int(array.min())<-(2**53) or int(array.max())>2**53):
            raise ValueError('Integer inputs must fit the exact float64 integer range')
    fills=fill_values.astype(np.float64,copy=True)
    matrix=values.astype(np.float64,copy=True)
    if not np.isfinite(fills).all() or np.isinf(matrix).any():raise ValueError('Finite fill values and non-infinite features required')
    converted=np.trunc(np.where(np.isnan(matrix),fills,matrix))
    bounds=np.iinfo(target)
    if converted.size and (int(converted.min())<bounds.min or int(converted.max())>bounds.max):
        raise ValueError('Truncated feature exceeds output integer range')
    return converted.astype(target)
