"""Fixed, ordered model inputs shared by training and inference."""
import numbers

import numpy as np
import pandas as pd

from sciona.fixed_feature_imputation import fixed_integer_features


def prepare_model_features(frame, feature_columns, numeric_fills, categorical_fills, *, integer_dtype='int16'):
    """Select an explicit feature order and apply complete disjoint fill policies.

    Numeric features use fixed per-column fills and checked truncation. Categories
    are strings or integral scalars, normalized to strings; missing values use the
    declared string fallback. No policy is inferred or fitted from the supplied
    rows. Metadata columns may remain in frame but never enter the returned
    model matrix. Original row order/index and input data are preserved.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('Feature DataFrame with unique columns required')
    if (not isinstance(feature_columns, list) or not feature_columns
            or not all(isinstance(c,str) and c for c in feature_columns)
            or len(set(feature_columns)) != len(feature_columns)
            or any(c not in frame for c in feature_columns)):
        raise ValueError('Explicit ordered unique available feature columns required')
    if not isinstance(numeric_fills, dict) or not isinstance(categorical_fills, dict):
        raise ValueError('Explicit numeric and categorical fill dictionaries required')
    if set(numeric_fills)&set(categorical_fills) or set(numeric_fills)|set(categorical_fills) != set(feature_columns):
        raise ValueError('Disjoint complete feature policies required')
    if not all(isinstance(v,numbers.Real) and not isinstance(v,(bool,np.bool_)) and np.isfinite(v) for v in numeric_fills.values()):
        raise ValueError('Finite numeric fill values required')
    if not all(isinstance(v,str) for v in categorical_fills.values()):
        raise ValueError('String categorical fills required')
    numeric=[c for c in feature_columns if c in numeric_fills]
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError('Numeric policy cannot implicitly parse string features')
        if pd.api.types.is_integer_dtype(frame[column]) and frame[column].notna().any():
            if int(frame[column].min()) < -(2**53) or int(frame[column].max()) > 2**53:
                raise ValueError('Integer features must fit exact float64 range')
    values=frame[numeric].to_numpy(dtype=np.float64,na_value=np.nan)
    converted=fixed_integer_features(values,np.array([numeric_fills[c] for c in numeric],dtype=np.float64),integer_dtype)
    outputs={column:converted[:,i] for i,column in enumerate(numeric)}
    for column,fallback in categorical_fills.items():
        normalized=[]
        for value in frame[column]:
            if isinstance(value,str):
                normalized.append(value)
            elif value is None or value is pd.NA or (isinstance(value,numbers.Real) and np.isnan(value)):
                normalized.append(fallback)
            elif isinstance(value,(numbers.Real,np.bool_)) and np.isfinite(value) and int(value)==value:
                normalized.append(str(int(value)))
            else:
                raise ValueError('Categorical values must be missing, strings or finite integral scalars')
        outputs[column]=np.asarray(normalized,dtype=object)
    return pd.DataFrame(outputs,index=frame.index.copy())[feature_columns]
