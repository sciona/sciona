"""Corrected event-delay features with explicit per-query availability."""
import numpy as np
from sciona.fixed_window_statistics import fixed_window_statistics


def event_delay_windows(event_ids, observed_at, actual_at, estimate_ids,
        estimate_observed_at, estimate_times, estimate_valid, query_times,
        lookbacks, rounding_step, ticks_per_output_unit):
    """Return count, valid-delay count, mean and maximum for each query/window.

    Integer times share one caller-defined unit. Events become available only
    when both observation and actual time are past or present. Window membership
    uses observation time rounded upward by rounding_step. The first nonmissing
    estimate observed by the query supplies the delay baseline; equal observation
    times use input order. Event rows are counted independently, without inferred
    deduplication. Delays use caller-defined output units. Missing estimates leave
    the event count intact and the delay missing. No source field names or policy
    constants are built in. Returns four (query, window) arrays.
    """
    for array in (observed_at,actual_at,estimate_observed_at,estimate_times,query_times,lookbacks):
        if not isinstance(array,np.ndarray) or array.dtype!=np.int64 or array.ndim!=1:
            raise ValueError('One-dimensional int64 time arrays required')
    for identifiers in (event_ids,estimate_ids):
        if not isinstance(identifiers,np.ndarray) or identifiers.ndim!=1 or not all(isinstance(v,str) and v for v in identifiers):
            raise ValueError('Nonempty string identifiers required')
    if event_ids.shape!=observed_at.shape or actual_at.shape!=observed_at.shape:
        raise ValueError('Aligned event rows required')
    if not isinstance(estimate_valid,np.ndarray) or estimate_valid.dtype!=np.bool_ or any(a.shape!=estimate_ids.shape for a in (estimate_valid,estimate_observed_at,estimate_times)):
        raise ValueError('Aligned estimates and explicit validity mask required')
    for value in (rounding_step,ticks_per_output_unit):
        if type(value) is not int or value<=0:raise ValueError('Positive integer time conversion controls required')
    if not len(lookbacks) or any(int(w)<=0 for w in lookbacks):raise ValueError('Positive windows required')
    rounded=[-((-int(t))//rounding_step)*rounding_step for t in observed_at]
    bounds=np.iinfo(np.int64)
    if any(t<bounds.min or t>bounds.max for t in rounded):raise ValueError('Rounded observation time exceeds int64')
    rounded=np.array(rounded,dtype=np.int64)
    available=np.maximum(observed_at,actual_at)
    first={}
    for index in sorted(range(len(estimate_ids)),key=lambda i:(int(estimate_observed_at[i]),i)):
        if estimate_valid[index]:first.setdefault(estimate_ids[index],index)
    outputs=[np.empty((len(query_times),len(lookbacks)),dtype=dtype) for dtype in (np.int64,np.int64,np.float64,np.float64)]
    for qi,query in enumerate(query_times):
        values=np.full(len(event_ids),np.nan);value_available=available.copy()
        for i,identity in enumerate(event_ids):
            index=first.get(identity)
            if index is not None and int(estimate_observed_at[index])<=int(query):
                values[i]=(int(actual_at[i])-int(estimate_times[index]))/ticks_per_output_unit
                value_available[i]=max(int(available[i]),int(estimate_observed_at[index]))
        statistics=fixed_window_statistics(rounded,available,value_available,values,query_times[qi:qi+1],lookbacks)
        for output,statistic in zip(outputs,statistics):output[qi]=statistic[0]
    return tuple(outputs)
