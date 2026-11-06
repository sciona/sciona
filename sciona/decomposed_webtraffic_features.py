"""Explicit feature blocks retaining the qualified Web Traffic numerical recipe.

Numeric correlation and popularity blocks have no page or calendar semantics.
Source arithmetic remains in the unchanged MIT-adapted numerical modules.
"""
import numpy as np
import pandas as pd
from .webtraffic_features import prepare_series,batch_autocorr
from .webtraffic_calendar import lag_indexes
from .webtraffic_pages import uniq_page_map,make_page_features,encode_page_features


def prepare(frame,valid_threshold,add_days,corr_backoffset):
    if not isinstance(frame,pd.DataFrame) or frame.empty:raise ValueError('Nonempty runtime series table required')
    if not isinstance(frame.columns,pd.DatetimeIndex) or frame.columns.tz is not None:
        raise ValueError('Timezone-naive daily columns required')
    if not frame.columns.equals(pd.date_range(frame.columns[0],periods=len(frame.columns))):
        raise ValueError('Contiguous increasing daily columns required')
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Unique sorted row labels required')
    for value in (add_days,corr_backoffset):
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<0:
            raise ValueError('Nonnegative integer offsets required')
    if len(frame.columns)+add_days>32768 or corr_backoffset>=len(frame.columns):
        raise ValueError('Offsets exceed source index range')
    logged,missing,starts,ends=prepare_series(frame,valid_threshold)
    if logged.empty:raise ValueError('No series survive validity filtering')
    return logged,missing,starts,ends


def correlations(values,starts,ends,lags,thresholds,backoffset):
    """Source weighted lag-neighbor autocorrelations with column normalization.

    The qualified source treats `ends` as an exclusive slice boundary despite
    obtaining it as the last positive position. Preserve that convention here.
    Undefined raw correlations become zero before population standardization;
    constant populations can produce NaN and are intentionally not repaired.
    """
    if not isinstance(values,np.ndarray) or values.ndim!=2 or not all(values.shape) or values.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty floating numeric panel required')
    for bounds in (starts,ends):
        if not isinstance(bounds,np.ndarray) or bounds.shape!=(len(values),) or bounds.dtype.kind not in 'iu' or np.any(bounds < -1) or np.any(bounds>=values.shape[1]):
            raise ValueError('Aligned integer support positions required')
    if np.any(starts>ends) or np.any((starts==-1)!=(ends==-1)):
        raise ValueError('Support positions differ')
    if not isinstance(lags,list) or not lags or any(type(lag) is not int or lag<2 for lag in lags):
        raise ValueError('Explicit integer lags of at least two required')
    if not isinstance(thresholds,list) or len(thresholds)!=len(lags) or any(type(t) not in (int,float) or not np.isfinite(t) or t<0 for t in thresholds):
        raise ValueError('Aligned finite nonnegative support thresholds required')
    if type(backoffset) is not int or not 0<=backoffset<values.shape[1]:raise ValueError('Valid integer backoffset required')
    columns=[]
    for lag,threshold in zip(lags,thresholds):
        raw=batch_autocorr(values,lag,starts,ends,threshold,backoffset)
        filled=np.nan_to_num(raw)
        with np.errstate(divide='ignore',invalid='ignore'):columns.append((filled-filled.mean())/np.std(filled))
    return np.stack(columns,axis=1)


def popularity(logged):
    """Row medians standardized across rows using pandas sample std (ddof=1)."""
    if not isinstance(logged,pd.DataFrame) or logged.empty or not np.isfinite(logged.to_numpy()).all():
        raise ValueError('Finite nonempty numeric frame required')
    medians=logged.median(axis=1)
    return (medians-medians.mean())/medians.std()


def page_metadata(row_labels):
    """Explicit source page/agent parsing policy; no inferred general grouping."""
    labels=np.asarray(row_labels,dtype=object)
    return uniq_page_map(labels),encode_page_features(make_page_features(labels))


def calendar(begin,end,add_days):
    """Source weekly cycle and fixed 3/6/9/12-month lag-index policy."""
    last=end+pd.Timedelta(add_days,unit='D')
    dates=pd.date_range(begin,last)
    phase=dates.dayofweek.values/(7/(2*np.pi))
    return np.stack([np.cos(phase),np.sin(phase)],axis=-1),np.stack(lag_indexes(begin,last),axis=-1),last


def assemble(logged,missing,correlation_columns,page_map,encoded,popularity_values,dow,lagged_ix,features_end):
    """Bind the source recipe order after separately computed feature blocks."""
    if correlation_columns.shape!=(len(logged),2):raise ValueError('Two source-ordered correlation columns required')
    if not missing.index.equals(logged.index) or not missing.columns.equals(logged.columns):
        raise ValueError('Missing mask must align exactly')
    if not popularity_values.index.equals(logged.index):raise ValueError('Popularity rows differ')
    if any(not value.index.equals(logged.index) for value in encoded.values()):raise ValueError('Encoded rows differ')
    if set(encoded)!={'agent','country','site'}:raise ValueError('Exact source category blocks required')
    count=len(pd.date_range(logged.columns[0],features_end))
    if dow.shape!=(count,2) or lagged_ix.shape!=(count,4):raise ValueError('Calendar blocks differ')
    restored=logged.copy();restored[missing]=np.nan
    tensors=dict(hits=restored,lagged_ix=lagged_ix,page_map=page_map,page_ix=logged.index.values,
        pf_agent=encoded['agent'],pf_country=encoded['country'],pf_site=encoded['site'],
        page_popularity=popularity_values,year_autocorr=correlation_columns[:,0],quarter_autocorr=correlation_columns[:,1],dow=dow)
    plain=dict(features_days=count,data_days=len(logged.columns),n_pages=len(logged),
        data_start=logged.columns[0],data_end=logged.columns[-1],features_end=features_end)
    return tensors,plain
