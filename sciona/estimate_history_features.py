"""Entity estimate histories with query-independent sample populations."""
import numpy as np


def estimate_history_features(entity_ids, observed_at, estimates, estimate_valid,
        query_ids, query_times, lookbacks, rounding_step, ticks_per_output_unit):
    """Compute remaining time, change from first estimate and rolling changes.

    Times share one integer clock. Observations become available at their ceiling
    grid boundary. Within each entity/bin, the latest nonmissing estimate wins
    (input order breaks equal-time ties); null bins carry the preceding estimate.
    Rolling populations contain history bins plus the current query if it is not
    already a bin. Other queries never become observations. This deliberately
    corrects source batch-dependent sample standard deviations. Windows are
    left-open/right-closed, sample standard deviation uses ddof=1, and missing
    numerical features remain NaN. Query order and duplicate queries survive.
    No domain identifiers, time units, grid spacing or lookbacks are implicit.
    """
    for times in (observed_at, estimates, query_times, lookbacks):
        if not isinstance(times, np.ndarray) or times.dtype != np.int64 or times.ndim != 1:
            raise ValueError('One-dimensional int64 time arrays required')
    for ids in (entity_ids, query_ids):
        if not isinstance(ids, np.ndarray) or ids.ndim != 1 or not all(isinstance(v, str) and v for v in ids):
            raise ValueError('Nonempty string entity identifiers required')
    if (not isinstance(estimate_valid, np.ndarray) or estimate_valid.dtype != np.bool_
            or any(a.shape != entity_ids.shape for a in (observed_at, estimates, estimate_valid))
            or query_ids.shape != query_times.shape):
        raise ValueError('Aligned histories and queries required')
    if not len(lookbacks) or any(int(w) <= 0 for w in lookbacks):
        raise ValueError('Positive lookbacks required')
    for control in (rounding_step, ticks_per_output_unit):
        if type(control) is not int or control <= 0:
            raise ValueError('Positive integer time controls required')
    bounds = np.iinfo(np.int64)
    histories = {}
    for i in sorted(range(len(entity_ids)), key=lambda i: (int(observed_at[i]), i)):
        tick = -((-int(observed_at[i])) // rounding_step) * rounding_step
        if tick < bounds.min or tick > bounds.max:
            raise ValueError('Rounded observation time exceeds int64')
        bins = histories.setdefault(entity_ids[i], {})
        bins.setdefault(tick, None)
        if estimate_valid[i]:
            bins[tick] = int(estimates[i])
    n = len(query_times)
    result = {key: np.full(n, np.nan) for key in ('remaining', 'change_from_first', 'change')}
    result.update({key: np.full((n, len(lookbacks)), np.nan) for key in ('sum', 'max', 'std')})
    result['estimate_available'] = np.zeros(n, dtype=bool)
    for qi, (entity, query) in enumerate(zip(query_ids, query_times)):
        query = int(query)
        bins = {t: value for t, value in histories.get(entity, {}).items() if t <= query}
        bins.setdefault(query, None)
        previous = None
        first = None
        ticks = []
        changes = []
        for t, value in sorted(bins.items()):
            current = previous if value is None else value
            delta = (0.0 if not ticks else np.nan) if current is None or previous is None else (current - previous) / ticks_per_output_unit
            if not ticks:
                delta = 0.0
            if first is None and current is not None:
                first = current
            ticks.append(t)
            changes.append(delta)
            previous = current
        result['change'][qi] = changes[-1]
        if previous is not None:
            result['estimate_available'][qi] = True
            result['remaining'][qi] = (previous - query) / ticks_per_output_unit
            result['change_from_first'][qi] = (previous - first) / ticks_per_output_unit
        for wi, width in enumerate(lookbacks):
            values = np.array([v for t, v in zip(ticks, changes) if query-int(width) < t <= query and not np.isnan(v)])
            if len(values):
                result['sum'][qi, wi] = values.sum()
                result['max'][qi, wi] = values.max()
                if len(values) > 1:
                    result['std'][qi, wi] = values.std(ddof=1)
    return result
