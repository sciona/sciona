"""Forecast features with distinct issue and valid-time availability."""
import numpy as np


def forecast_history_features(issued_at, valid_at, values, query_times, *,
        history_window, lead_bands, contrast_leads, revision_lags):
    """Return generic historical, lead-band, revision and contrast features.

    Integer times and lead controls share one caller-defined clock. Only rows
    issued by a query participate. (issue, valid) pairs must be unique. Historical
    mean/min/max use the first nonmissing forecast per field from each issue in
    (query-history_window, query]. Lead-band means and paired lead contrasts use
    only the latest available issue. Bands are (lower, upper], with None meaning
    no lower bound. Latest-valid values use the most recent valid time <= query
    and its latest available issue; revisions subtract earlier issues for that
    same valid time by explicit row lags. No future issues, future valid times
    for historical revisions, inferred imputation or cross-issue horizon fills.
    NaN represents unavailable output; no domain names or time scales are built in.
    """
    for times in (issued_at, valid_at, query_times, revision_lags):
        if not isinstance(times, np.ndarray) or times.dtype != np.int64 or times.ndim != 1:
            raise ValueError('One-dimensional int64 time and lag arrays required')
    if (not isinstance(values, np.ndarray) or values.dtype != np.float64 or values.ndim != 2
            or values.shape[0] != len(issued_at) or valid_at.shape != issued_at.shape
            or np.isinf(values).any()):
        raise ValueError('Aligned finite-or-missing float64 forecast matrix required')
    if len(set(zip(issued_at.tolist(), valid_at.tolist()))) != len(issued_at):
        raise ValueError('Unique issue/valid-time pairs required')
    if type(history_window) is not int or history_window <= 0:
        raise ValueError('Positive integer history window required')
    if not isinstance(lead_bands, list) or not lead_bands:
        raise ValueError('Explicit nonempty lead bands required')
    for band in lead_bands:
        if (not isinstance(band, tuple) or len(band) != 2 or type(band[1]) is not int
                or (band[0] is not None and (type(band[0]) is not int or band[0] >= band[1]))):
            raise ValueError('Lead bands must be (integer-or-None, integer]')
    if (not isinstance(contrast_leads, np.ndarray) or contrast_leads.dtype != np.int64
            or contrast_leads.ndim != 2 or contrast_leads.shape[1] != 2):
        raise ValueError('Int64 paired contrast leads required')
    if not len(revision_lags) or any(int(lag) <= 0 for lag in revision_lags):
        raise ValueError('Positive revision lags required')
    n, fields = len(query_times), values.shape[1]
    result = {key: np.full((n, fields), np.nan) for key in ('mean', 'min', 'max', 'latest_valid')}
    result['bands'] = np.full((n, len(lead_bands), fields), np.nan)
    result['contrasts'] = np.full((n, len(contrast_leads), fields), np.nan)
    result['revisions'] = np.full((n, len(revision_lags), fields), np.nan)
    result['issue_available'] = np.zeros(n, dtype=bool)
    order = sorted(range(len(issued_at)), key=lambda i: (int(issued_at[i]), int(valid_at[i])))
    for qi, q in enumerate(query_times):
        query = int(q)
        available = [i for i in order if int(issued_at[i]) <= query]
        if not available:
            continue
        result['issue_available'][qi] = True
        issues = sorted({int(issued_at[i]) for i in available})
        snapshots = []
        for issue in issues:
            if query-history_window < issue <= query:
                rows = [i for i in available if int(issued_at[i]) == issue]
                snapshot = np.full(fields, np.nan)
                for field in range(fields):
                    nonmissing = [values[i, field] for i in rows if not np.isnan(values[i, field])]
                    if nonmissing:
                        snapshot[field] = nonmissing[0]
                snapshots.append(snapshot)
        for field in range(fields):
            samples = np.array([row[field] for row in snapshots if not np.isnan(row[field])])
            if len(samples):
                result['mean'][qi, field] = samples.mean()
                result['min'][qi, field] = samples.min()
                result['max'][qi, field] = samples.max()
        latest_issue = issues[-1]
        rows = [i for i in available if int(issued_at[i]) == latest_issue]
        by_lead = {int(valid_at[i])-latest_issue: i for i in rows}
        for bi, (lower, upper) in enumerate(lead_bands):
            selected = [i for lead, i in by_lead.items() if (lower is None or lead > lower) and lead <= upper]
            for field in range(fields):
                samples = [values[i, field] for i in selected if not np.isnan(values[i, field])]
                if samples:
                    result['bands'][qi, bi, field] = np.mean(samples)
        for ci, (left, right) in enumerate(contrast_leads):
            if int(left) in by_lead and int(right) in by_lead:
                result['contrasts'][qi, ci] = values[by_lead[int(left)]]-values[by_lead[int(right)]]
        past = [i for i in available if int(valid_at[i]) <= query]
        if past:
            latest_valid = max(int(valid_at[i]) for i in past)
            revisions = [i for i in past if int(valid_at[i]) == latest_valid]
            result['latest_valid'][qi] = values[revisions[-1]]
            for li, lag in enumerate(revision_lags):
                if len(revisions) > int(lag):
                    result['revisions'][qi, li] = values[revisions[-1]]-values[revisions[-1-int(lag)]]
    if any(np.isinf(value).any() for value in result.values()):
        raise ValueError('Forecast arithmetic outside finite range')
    return result
