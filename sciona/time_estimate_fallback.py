"""Positional fallback from latest nonmissing grouped time estimates."""
import numpy as np
import pandas as pd


def grouped_time_estimate_fallback(query, history, *, group_column, order_column,
        estimate_column, query_time_column, seconds_per_unit, offset, lower, upper, missing):
    """Return one float prediction per query row without changing caller tables.

    Estimates and query times must support pandas datetime subtraction. All
    numeric settings share the caller's output units, except seconds_per_unit.
    Source grouping uses last nonmissing value after quicksort, not last row.
    Equal ordering keys have no stable-order guarantee. Unknown groups use
    `missing`; no cross-group imputation is performed. Caller indices are ignored.
    """
    if not isinstance(query,pd.DataFrame) or not isinstance(history,pd.DataFrame):
        raise ValueError('Query and history DataFrames required')
    if not query.columns.is_unique or not history.columns.is_unique:
        raise ValueError('Unique columns required')
    if len({group_column,order_column,estimate_column})!=3 or group_column==query_time_column or estimate_column==query_time_column:
        raise ValueError('Distinct grouping, estimate and time roles required')
    if any(c not in query for c in [group_column,query_time_column]) or any(c not in history for c in [group_column,order_column,estimate_column]):
        raise ValueError('Required role columns missing')
    settings=[seconds_per_unit,offset,lower,upper,missing]
    if any(type(x) not in (int,float) or not np.isfinite(x) for x in settings) or seconds_per_unit<=0 or not lower<=missing<=upper:
        raise ValueError('Finite consistent unit conversion and fallback bounds required')
    latest=history.sort_values(order_column,kind='quicksort').groupby(group_column).last()[estimate_column]
    left=query[[group_column,query_time_column]].reset_index(drop=True)
    merged=left.merge(latest,how='left',on=group_column,sort=False,validate='many_to_one')
    values=(merged[estimate_column]-merged[query_time_column]).dt.total_seconds()/seconds_per_unit-offset
    return values.clip(lower=lower,upper=upper).fillna(missing).to_numpy(dtype=np.float64,copy=True)
