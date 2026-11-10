"""Branch-specific source metadata and longitudinal target preparation."""
import numpy as np
import pandas as pd

from sciona.affine_trajectory_fit import fit


def encode(numeric, category, status, *, branch):
    """Encode one source record, preserving training/inference case asymmetry.

The numeric field uses the source's (value-30)/30 transform. Category/status
interpretation is domain policy, not a generic categorization operation.
Unrecognized strings retain the source fallback; missing values are rejected.
"""
    value=np.asarray(numeric)
    if (value.ndim!=0 or value.dtype.kind not in 'iuf' or not np.isfinite(value)
            or not isinstance(category,str) or not isinstance(status,str)):
        raise ValueError('Finite real numeric value and category strings required')
    if branch not in ('training','inference'):
        raise ValueError('Explicit training or inference branch required')
    comparison=category.lower() if branch=='training' else category
    bits={'Never smoked':[0,0],'Ex-smoker':[1,1],'Currently smokes':[0,1]}.get(status,[1,0])
    result=np.array([(value[()]-30)/30,0 if comparison=='male' else 1,*bits])
    if not np.isfinite(result).all():
        raise ValueError('Finite encoded metadata required')
    return result


def prepare_training(table, *, time_coordinate, response_coordinate):
    """Preserve first-encounter entity order and each entity's first metadata row.

The caller maps raw fields to this neutral schema and establishes observation
availability. Slopes are in response-coordinate units per time-coordinate unit.
The explicit -1 cutoff reconstructs the documented pre-NumPy-2 default; it
does not claim historical LAPACK bitwise equivalence. No records are excluded.
"""
    columns={'entity','time','target','numeric','category_a','category_b'}
    if (not isinstance(table,pd.DataFrame) or not table.columns.is_unique
            or set(table.columns)!=columns or table.empty):
        raise ValueError('Nonempty explicitly mapped training table required')
    if not all(isinstance(v,str) and v for v in (time_coordinate,response_coordinate)):
        raise ValueError('Explicit time and response coordinates required')
    if not all(isinstance(v,str) and v for v in table['entity']):
        raise ValueError('Nonempty string entity identities required')
    keys=list(table['entity'].unique())
    metadata=[];slopes=[];diagnostics=[]
    for key in keys:
        group=table.loc[table['entity']==key]
        fitted=fit(group['time'].to_numpy(),group['target'].to_numpy(),rcond=-1)
        first=group.iloc[0]
        metadata.append(encode(first['numeric'],first['category_a'],first['category_b'],branch='training'))
        slopes.append(fitted['coefficients'][0])
        diagnostics.append(fitted)
    return dict(entities=keys,metadata=np.array(metadata),slopes=np.array(slopes),
                fits=diagnostics,time_coordinate=time_coordinate,response_coordinate=response_coordinate)
