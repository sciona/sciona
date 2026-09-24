"""Reusable estimate fallback with explicit per-query observation availability."""
import numpy as np

from sciona.asof_indices import latest_available_indices


def asof_estimate_fallback(entity_ids, observed_at, estimates, estimate_valid,
        query_ids, query_times, *, ticks_per_unit, offset, lower, upper, missing):
    """Latest nonmissing observed estimate minus query time, offset and bounds.

    Integer times share one explicit clock. Equal observation timestamps use the
    last input row; null estimates do not erase earlier nonmissing observations.
    Future observations never contribute. Query order and duplicates survive.
    Bounds, offset and missing fallback use the caller's output units. Fractional
    results remain fractional. Missing groups use the explicit fallback directly.
    """
    for ids in (entity_ids,query_ids):
        if not isinstance(ids,np.ndarray) or ids.ndim!=1 or not all(isinstance(v,str) and v for v in ids):
            raise ValueError('Nonempty string entity identifiers required')
    for times in (observed_at,estimates,query_times):
        if not isinstance(times,np.ndarray) or times.dtype!=np.int64 or times.ndim!=1:
            raise ValueError('One-dimensional int64 time arrays required')
    if (not isinstance(estimate_valid,np.ndarray) or estimate_valid.dtype!=np.bool_
            or any(a.shape!=entity_ids.shape for a in (observed_at,estimates,estimate_valid))
            or query_ids.shape!=query_times.shape):
        raise ValueError('Aligned histories and queries required')
    controls=[ticks_per_unit,offset,lower,upper,missing]
    if any(type(v) not in (int,float) or not np.isfinite(v) for v in controls) or ticks_per_unit<=0 or not lower<=missing<=upper:
        raise ValueError('Finite consistent units, bounds and fallback required')
    result=np.full(len(query_ids),float(missing))
    for identity in set(query_ids):
        rows=np.flatnonzero((entity_ids==identity)&estimate_valid)
        queries=np.flatnonzero(query_ids==identity)
        indices,available=latest_available_indices(observed_at[rows],query_times[queries])
        for qi,index,valid in zip(queries,indices,available):
            if valid:
                value=(int(estimates[rows[index]])-int(query_times[qi]))/ticks_per_unit-offset
                if not np.isfinite(value):raise ValueError('Fallback arithmetic outside finite range')
                result[qi]=min(max(value,lower),upper)
    return result
