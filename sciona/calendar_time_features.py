"""Calendar features with an explicit caller-supplied holiday schedule."""
import numpy as np
import pandas as pd


def calendar_time_features(query_times, holiday_midnights):
    """Return row-aligned calendar arrays for timezone-naive timestamps.

    Holidays must be sorted, unique midnight values. Holiday membership compares
    dates; countdown selects the next holiday timestamp at or after the query
    and floors elapsed days. Thus midday on a holiday is flagged as a holiday,
    but counts toward the next holiday midnight. This preserves the qualified
    source convention. Missing calendar coverage raises rather than inventing a
    holiday. Locale, holiday rules and clock interpretation belong to the caller.
    """
    for times in (query_times,holiday_midnights):
        if not isinstance(times,pd.DatetimeIndex) or times.tz is not None or times.hasnans:
            raise ValueError('Nonmissing timezone-naive DatetimeIndex required')
    holidays=holiday_midnights
    if not len(holidays) or not holidays.is_unique or not holidays.is_monotonic_increasing or not holidays.equals(holidays.normalize()):
        raise ValueError('Sorted unique holiday midnights required')
    indices=holidays.searchsorted(query_times,side='left')
    if np.any(indices==len(holidays)):raise ValueError('Holiday schedule does not cover query horizon')
    # Timestamp subtraction checks duration overflow instead of wrapping int64.
    days=np.array([(holidays[i]-q).days for q,i in zip(query_times,indices)],dtype=np.int64)
    return dict(hour=query_times.hour.to_numpy(),weekday=query_times.dayofweek.to_numpy(),
        month=query_times.month.to_numpy(),day=query_times.day.to_numpy(),
        iso_week=query_times.isocalendar().week.to_numpy(dtype=np.int64),
        week_of_month=((query_times.day.to_numpy()-1)//7+1),
        is_holiday=np.asarray(query_times.normalize().isin(holidays)),days_until_holiday=days)
