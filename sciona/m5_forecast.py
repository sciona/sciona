"""Independent complete M5 family forecasts and six-way arithmetic mean."""
import numpy as np
from sciona.m5_family import frame,pools
from sciona.m5_lags import rolling


def predict_families(prepared, models, cutoff):
    grid=prepared['grid'];days=np.asarray(grid['day']);series=np.asarray(grid['series'])
    if isinstance(cutoff,(bool,np.bool_)) or not isinstance(cutoff,(int,np.integer)):
        raise ValueError('Cutoff must be an integer')
    identities=np.unique(series)
    future=(days>cutoff)&(days<=cutoff+28)
    expected={(int(s),int(cutoff)+d) for s in identities for d in range(1,29)}
    actual=list(zip(map(int,series[future]),map(int,days[future])))
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Every series needs exactly 28 aligned future rows')
    if not np.isnan(grid['target'][future]).all():
        raise ValueError('Future targets must be missing')
    family_keys=[(recursive,pooling) for recursive in (True,False)
                 for pooling in ('outlet','outlet_category','outlet_department')]
    if {(m.recursive,m.pooling) for m in models}!=set(family_keys):
        raise ValueError('All six model families required')
    forecasts=[]
    for recursive,pooling in family_keys:
        selected=[m for m in models if (m.recursive,m.pooling)==(recursive,pooling)]
        expected_pools=pools(prepared,pooling)
        if len(selected)!=len(expected_pools) or {m.pool for m in selected}!=set(expected_pools):
            raise ValueError('Exact unique family pool inventory required')
        # Each recursive family gets its own forecast feedback. Historical
        # values already carry preprocessing quantization; learned predictions
        # retain float64, matching source promotion on nonrepresentable writes.
        state=np.asarray(grid['target'],dtype=np.float64).copy()
        state[days>cutoff]=np.nan
        retained=(days>cutoff-100)&(days<=cutoff+28)
        row_forecast=np.full(len(days),np.nan)
        for horizon in (range(1,29) if recursive else (None,)):
            temporary={}
            if recursive:
                for shift in (1,7,14):
                    for window in (7,14,30,60):
                        values=np.full(len(days),np.nan)
                        values[retained]=rolling(state[retained],series[retained],shift,window)
                        temporary[f'temporary_{shift}_{window}']=values
            assignments=np.zeros(len(days),dtype=np.int8)
            for item in selected:
                # Preserve training column precision for nonrecursive inference.
                first_day=int(days[item.rows].min())
                features,rows=frame(prepared,recursive=recursive,pooling=pooling,pool=item.pool,first_day=first_day)
                if tuple(features.columns)!=item.features:
                    raise ValueError('Forecast/model feature order differs')
                for name,values in temporary.items():features[name]=values[rows]
                mask=(days[rows]==cutoff+horizon) if recursive else future[rows]
                target_rows=rows[mask]
                if not target_rows.size:raise ValueError('Empty forecast pool')
                values=np.asarray(item.model.predict(features.loc[mask]),dtype=np.float64)
                if values.shape!=(len(target_rows),) or not np.isfinite(values).all() or (values<0).any():
                    raise ValueError('Invalid model forecast')
                row_forecast[target_rows]=values
                state[target_rows]=values
                assignments[target_rows]+=1
            wanted=(days==cutoff+horizon) if recursive else future
            if not np.array_equal(assignments,wanted.astype(np.int8)):
                raise ValueError('Forecast rows must be assigned exactly once')
        position={int(s):i for i,s in enumerate(identities)}
        matrix=np.full((len(identities),28),np.nan)
        for row in np.flatnonzero(future):
            matrix[position[int(series[row])],int(days[row]-cutoff-1)]=row_forecast[row]
        if not np.isfinite(matrix).all():raise ValueError('Incomplete family forecast')
        forecasts.append(matrix)
    stacked=np.stack(forecasts)
    return dict(families=stacked,models=len(models),horizon=28,
                scope='Independent CPU full-horizon family mean; no historical accuracy or GPU parity claim.')


def predict(prepared, models, cutoff):
    result = predict_families(prepared, models, cutoff)
    return dict(forecast=np.sum(result['families'], axis=0)/6, **result)
