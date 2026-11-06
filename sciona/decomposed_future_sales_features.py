"""Explicit numerical stages for the complete selected forecasting feature branch.

No provider is registered here. Rolling computation is an explicit callable
input so a CDG can bind the existing rolling atom instead of hiding or copying it.
"""
import math
import numpy as np
from sciona.future_sales_features import Panel


def aggregate(transactions, entities, boundaries):
    """Net finite transactions within caller periods, retaining signed totals."""
    if not isinstance(boundaries,list) or len(boundaries)<4 or any(type(v) is not int for v in boundaries) or any(b<=a for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('Strictly increasing integer period boundaries required')
    if not isinstance(entities,list) or not entities:raise ValueError('Explicit entity population required')
    keys=[];lookup={};categories={}
    for e in entities:
        if not isinstance(e,dict) or set(e)!={'store','item','category'} or any(type(v) is not int or not 0<=v<2**31 for v in e.values()):raise ValueError('Nonnegative bounded integer entity slots required')
        key=e['store'],e['item']
        if key in lookup:raise ValueError('Duplicate entity')
        if e['item'] in categories and categories[e['item']]!=e['category']:raise ValueError('Inconsistent item category')
        categories[e['item']]=e['category'];lookup[key]=len(keys);keys.append((e['store'],e['item'],e['category']))
    if not isinstance(transactions,list):raise ValueError('Explicit transaction list required')
    amounts={}
    for t in transactions:
        if not isinstance(t,dict) or set(t)!={'day','store','item','quantity'}:raise ValueError('Exact transaction fields required')
        if any(type(t[k]) is not int for k in ('day','store','item')) or type(t['quantity']) not in (int,float) or not math.isfinite(t['quantity']):raise ValueError('Finite numeric transaction required')
        if not boundaries[0]<=t['day']<boundaries[-2]:raise ValueError('Transactions must precede the forecast period')
        key=t['store'],t['item']
        if key not in lookup:raise ValueError('Transaction outside explicit entity population')
        period=int(np.searchsorted(boundaries,t['day'],side='right')-1)
        amounts.setdefault((period,lookup[key]),[]).append(t['quantity'])
    totals=np.zeros((len(boundaries)-2,len(keys)),dtype=np.float64)
    for key,values in amounts.items():
        try:value=math.fsum(values)
        except OverflowError:raise ValueError('Aggregate exceeds finite range') from None
        if not math.isfinite(value):raise ValueError('Aggregate exceeds finite range')
        totals[key]=value
    return totals,np.asarray(keys,dtype=np.int64)


def bounded_targets(totals, lower, upper):
    """Clip observed numeric panels and append exactly one unknown future row."""
    if not isinstance(totals,np.ndarray) or totals.dtype!=np.float64 or totals.ndim!=2 or min(totals.shape)<1 or not np.isfinite(totals).all():
        raise ValueError('Nonempty finite float64 observed panel required')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (lower,upper)) or lower>=upper:
        raise ValueError('Finite ordered clipping bounds required')
    return np.concatenate((np.clip(totals,lower,upper),np.full((1,totals.shape[1]),np.nan)),axis=0)


def _targets(targets):
    if not isinstance(targets,np.ndarray) or targets.dtype!=np.float64 or targets.ndim!=2 or targets.shape[0]<2 or targets.shape[1]<1:
        raise ValueError('Observed and future float64 panel required')
    if not np.isfinite(targets[:-1]).all() or not np.isnan(targets[-1]).all():
        raise ValueError('Finite observed rows and one unknown future row required')


def lag_features(targets, lags, rolling_windows, rolling_function):
    """Prior-only lags and full-width means using an externally bound rolling op."""
    _targets(targets)
    for offsets in (lags,rolling_windows):
        if not isinstance(offsets,list) or not offsets or any(type(x) is not int or x<1 for x in offsets) or len(set(offsets))!=len(offsets):
            raise ValueError('Unique positive lag/window offsets required')
    if not callable(rolling_function):raise ValueError('Explicit rolling callable required')
    periods,entities=targets.shape
    result=np.full((periods,entities,len(lags)+len(rolling_windows)),np.nan,dtype=np.float64)
    for column,offset in enumerate(lags):
        if offset<periods:result[offset:,:,column]=targets[:-offset]
    for column,width in enumerate(rolling_windows,start=len(lags)):
        for entity in range(entities):
            # Copy prevents a supplied callable from mutating caller state.
            values=rolling_function(targets[:-1,entity].copy(),width,['mean'])
            if not isinstance(values,np.ndarray) or values.dtype!=np.float64 or values.shape!=(periods-1,1) or not np.isfinite(values).all():
                raise ValueError('Rolling provider returned incompatible values')
            if width<periods:result[width:,entity,column]=values[width-1:,0]
    return result.reshape(periods*entities,-1)


def encoding_features(targets, groups, calendar_cycle, cold_start):
    """Numeric group encodings independent of store/item/category names.

    Preserve prior-only means and exact reduction ordering. Every group column
    defines one partition, with any positive number of grouping columns.
    """
    _targets(targets)
    if not isinstance(groups,np.ndarray) or groups.dtype!=np.int64 or groups.ndim!=2 or groups.shape[0]!=targets.shape[1] or groups.shape[1]<1 or (groups<0).any():
        raise ValueError('Aligned nonnegative int64 entity group matrix required')
    if type(calendar_cycle) is not int or calendar_cycle<1 or type(cold_start) not in (int,float) or not math.isfinite(cold_start):
        raise ValueError('Positive cycle and finite cold-start value required')
    result=[]
    for period in range(targets.shape[0]):
        history=targets[:period]
        for index,group in enumerate(groups):
            values=[*group,period,math.sin(2*math.pi*(period%calendar_cycle)/calendar_cycle),math.cos(2*math.pi*(period%calendar_cycle)/calendar_cycle)]
            if period:
                means=[float(history[:,index].mean())]+[float(history[:,groups[:,axis]==value].mean()) for axis,value in enumerate(group)]+[float(history.mean())]
            else:means=[float(cold_start)]*(groups.shape[1]+2)
            result.append(values+means)
    result=np.asarray(result,dtype=np.float64)
    if not np.isfinite(result).all():raise ValueError('Encoding feature overflow')
    return result


def assemble(targets, lagged, encoded, lags, rolling_windows):
    """Apply the retained three-category feature layout at the recipe boundary."""
    _targets(targets)
    rows=targets.size
    for value,width in ((lagged,len(lags)+len(rolling_windows)),(encoded,11)):
        if not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.shape!=(rows,width) or np.isinf(value).any():
            raise ValueError('Aligned feature blocks required')
    if not np.isfinite(encoded).all():raise ValueError('Finite encodings required')
    features=np.concatenate((encoded[:,:6],lagged,encoded[:,6:]),axis=1)
    names=('store_slot','item_slot','category_slot','period_index','cycle_sin','cycle_cos')+tuple('lag_'+str(v) for v in lags)+tuple('rolling_'+str(v) for v in rolling_windows)+('entity_expanding','store_expanding','item_expanding','category_expanding','global_expanding')
    return Panel(features,targets.reshape(-1).copy(),np.repeat(np.arange(targets.shape[0]),targets.shape[1]),targets.shape[1],names)
