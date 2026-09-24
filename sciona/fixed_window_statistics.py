"""Fixed past windows with separate event and value observation availability."""
import numpy as np


def fixed_window_statistics(event_times, event_available_at, value_available_at,
        values, query_times, lookbacks):
    """Return event count, value count, mean and maximum in (query-width, query].

    Times, availability times, queries and positive lookbacks use one explicit
    caller-defined integer time unit. Each supplied row is one countable event;
    event availability should include both observation and actual-event time.
    A value contributes only if its event and value are available by the query.
    NaNs mean missing values; infinities are rejected. Empty value populations
    yield NaN statistics, while event and value counts remain independently valid.
    No grouping, deduplication, time rounding or widening is inferred. Outputs
    have shape (number of queries, number of lookbacks). Inputs are not mutated.
    """
    for array in (event_times,event_available_at,value_available_at,query_times,lookbacks):
        if not isinstance(array,np.ndarray) or array.dtype!=np.int64 or array.ndim!=1:
            raise ValueError('One-dimensional int64 time arrays required')
    if not isinstance(values,np.ndarray) or values.dtype!=np.float64 or values.ndim!=1:
        raise ValueError('One-dimensional float64 values required')
    if any(a.shape!=event_times.shape for a in (event_available_at,value_available_at,values)):
        raise ValueError('Aligned event, availability and value arrays required')
    if not len(lookbacks) or any(int(w)<=0 for w in lookbacks) or np.isinf(values).any():
        raise ValueError('Positive lookbacks and finite-or-missing values required')
    shape=(len(query_times),len(lookbacks))
    counts=np.zeros(shape,dtype=np.int64);value_counts=np.zeros(shape,dtype=np.int64)
    means=np.full(shape,np.nan);maxima=np.full(shape,np.nan)
    for i,q in enumerate(query_times):
        query=int(q)
        for j,w in enumerate(lookbacks):
            low=query-int(w)  # Python integers prevent int64 boundary overflow.
            events=np.fromiter((low<int(t)<=query and int(a)<=query for t,a in zip(event_times,event_available_at)),dtype=bool,count=len(event_times))
            selected=events&(value_available_at<=q)&~np.isnan(values)
            counts[i,j]=np.count_nonzero(events);value_counts[i,j]=np.count_nonzero(selected)
            if value_counts[i,j]:
                with np.errstate(over='ignore',invalid='ignore'):
                    mean=float(np.mean(values[selected]));maximum=float(np.max(values[selected]))
                if not np.isfinite(mean):raise ValueError('Mean outside supported finite arithmetic range')
                means[i,j]=mean;maxima[i,j]=maximum
    return counts,value_counts,means,maxima
