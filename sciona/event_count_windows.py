"""Availability-aware counts for explicitly selected event channels."""
import numpy as np

from sciona.fixed_window_statistics import fixed_window_statistics


def event_count_windows(observed_at, actual_at, countable, query_times, lookbacks):
    """Return (query, channel, window) counts in (query-width, query].

    All times use one caller-defined integer clock. Observation times determine
    window membership. Each channel has an actual-event time and a countable
    mask; missing identifiers, categories or actual times must be masked out by
    the caller. Masked actual-time placeholders have no effect. A countable row
    becomes available only after both its observation and actual event time.
    Rows count independently: deduplication is not inferred. Query order and
    duplicates are preserved, and tied observations are order-independent.
    """
    for array in (observed_at, query_times, lookbacks):
        if not isinstance(array, np.ndarray) or array.dtype != np.int64 or array.ndim != 1:
            raise ValueError('One-dimensional int64 time arrays required')
    if (not isinstance(actual_at, np.ndarray) or actual_at.dtype != np.int64
            or actual_at.ndim != 2 or actual_at.shape[0] != len(observed_at)):
        raise ValueError('Aligned int64 event-channel actual times required')
    if (not isinstance(countable, np.ndarray) or countable.dtype != np.bool_
            or countable.shape != actual_at.shape):
        raise ValueError('Aligned boolean event-channel countability mask required')
    if not len(lookbacks) or any(int(width) <= 0 for width in lookbacks):
        raise ValueError('Positive lookbacks required')
    result = np.zeros((len(query_times), actual_at.shape[1], len(lookbacks)), dtype=np.int64)
    for channel in range(actual_at.shape[1]):
        selected = countable[:, channel]
        times = observed_at[selected]
        available = np.maximum(times, actual_at[selected, channel])
        result[:, channel] = fixed_window_statistics(
            times, available, available, np.zeros(len(times), dtype=np.float64),
            query_times, lookbacks)[0]
    return result
