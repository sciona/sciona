"""Reusable categorical state features on an explicit regular query grid."""
import numpy as np
from sciona.asof_indices import latest_available_indices


def configuration_state_features(observed_at, states, query_times, vocabulary,
        suffixes, change_windows, grid_step):
    """Encode latest available string states, counts and rolling changes.

    Times use one integer clock; queries form a strictly increasing regular grid.
    States is (observation, channel), with no missing strings. Vocabulary and
    suffix priority are explicit: absent token -> 0, first matching suffix ->
    1..K, otherwise K+1. Matching is literal substring matching. Item counts use
    comma count plus one, including the source convention for empty strings.
    Unavailable state outputs are placeholders marked by state_valid=False.
    Changes start only across two available adjacent states. Full-window change
    sums are zero until enough available grid rows exist, with window_valid=False.
    Caller must apply masks before treating unavailable features as observations.
    """
    indices,valid=latest_available_indices(observed_at,query_times)
    if type(grid_step) is not int or grid_step<=0 or any(int(b)-int(a)!=grid_step for a,b in zip(query_times,query_times[1:])):
        raise ValueError('Explicit regular increasing query grid required')
    if not isinstance(states,np.ndarray) or states.ndim!=2 or len(states)!=len(observed_at) or not all(isinstance(v,str) for v in states.flat):
        raise ValueError('Aligned string state matrix required')
    for tokens in (vocabulary,suffixes):
        if not isinstance(tokens,list) or not tokens or not all(isinstance(v,str) and v for v in tokens) or len(set(tokens))!=len(tokens):
            raise ValueError('Unique ordered nonempty token lists required')
    if not isinstance(change_windows,np.ndarray) or change_windows.dtype!=np.int64 or change_windows.ndim!=1 or not len(change_windows) or any(int(w)<=0 for w in change_windows):
        raise ValueError('Positive int64 grid-row window sizes required')
    n=len(query_times);channels=states.shape[1]
    encoded=np.zeros((n,channels,len(vocabulary)),dtype=np.int64)
    counts=np.zeros((n,channels),dtype=np.int64);changes=np.zeros((n,channels),dtype=bool)
    for i in range(n):
        if not valid[i]:continue
        for j,text in enumerate(states[indices[i]]):
            counts[i,j]=text.count(',')+1
            for k,token in enumerate(vocabulary):
                if token in text:
                    encoded[i,j,k]=next((p+1 for p,suffix in enumerate(suffixes) if token+suffix in text),len(suffixes)+1)
            if i and valid[i-1]:changes[i,j]=text!=states[indices[i-1],j]
    rolling=np.zeros((n,channels,len(change_windows)),dtype=np.int64)
    complete=np.zeros((n,len(change_windows)),dtype=bool)
    for i in range(n):
        for j,width in enumerate(change_windows):
            start=i-int(width)+1
            if start>=0 and valid[start:i+1].all():
                complete[i,j]=True;rolling[i,:,j]=changes[start:i+1].sum(axis=0)
    return dict(categories=encoded,item_counts=counts,changes=changes,rolling_changes=rolling,state_valid=valid,window_valid=complete)
