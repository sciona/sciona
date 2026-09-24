"""Reusable positional lookup of observations available at query time."""
import numpy as np


def latest_available_indices(observed_at, query_times):
    """Return original row indices and availability mask for observation <= query.

    Inputs are one-dimensional int64 arrays sharing one caller-defined clock and
    time unit. Input order need not be sorted. Equal observation times select
    the last input row. Queries without a prior observation return index -1 and
    mask False; callers must use the mask before indexing values. No future
    fill, grouping, expiration or default-value policy is inferred.
    """
    for array in (observed_at,query_times):
        if not isinstance(array,np.ndarray) or array.dtype!=np.int64 or array.ndim!=1:
            raise ValueError('One-dimensional int64 times required')
    order=np.argsort(observed_at,kind='stable')
    positions=np.searchsorted(observed_at[order],query_times,side='right')-1
    available=positions>=0
    indices=np.full(query_times.shape,-1,dtype=np.int64)
    indices[available]=order[positions[available]]
    return indices,available
