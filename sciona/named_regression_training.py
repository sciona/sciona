"""Reusable temporal splits and bounded native training for named feature tables."""
import numpy as np
import pandas as pd


def temporal_regression_split(frame, targets, times, eligible, cutoff, offsets, categorical_columns):
    """Split eligible rows at a declared cutoff, subtracting explicit target offsets.

    Row order and frame index are retained. Inputs share one naive clock; target
    and offset units must agree. Categories are explicit column names, never
    inferred from a domain naming convention. The cutoff belongs to validation.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not len(frame.columns):
        raise ValueError('Nonempty unique named feature schema required')
    count = len(frame)
    target = np.asarray(targets, dtype=np.float64)
    offset = np.asarray(offsets, dtype=np.float64)
    mask = np.asarray(eligible)
    if target.shape != (count,) or offset.shape != (count,) or mask.shape != (count,) or mask.dtype.kind != 'b':
        raise ValueError('Aligned targets, offsets and boolean eligibility required')
    clock = pd.DatetimeIndex(times)
    boundary = pd.Timestamp(cutoff)
    if len(clock) != count or clock.hasnans or clock.tz is not None or pd.isna(boundary) or boundary.tz is not None:
        raise ValueError('Aligned nonmissing naive times and cutoff required')
    if not isinstance(categorical_columns, list) or len(set(categorical_columns)) != len(categorical_columns) or any(c not in frame for c in categorical_columns):
        raise ValueError('Unique explicit categorical columns required')
    if not np.isfinite(target[mask]).all() or not np.isfinite(offset[mask]).all():
        raise ValueError('Finite eligible targets and offsets required')
    train = mask & (clock < boundary)
    valid = mask & (clock >= boundary)
    if not train.any() or not valid.any():
        raise ValueError('Nonempty eligible training and validation required')
    with np.errstate(over='raise', invalid='raise'):
        adjusted = target[mask] - offset[mask]
    y = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    y.iloc[np.flatnonzero(mask)] = adjusted
    categories = [frame.columns.get_loc(c) for c in categorical_columns]
    return frame.iloc[np.flatnonzero(train)].copy(), y.iloc[np.flatnonzero(train)].copy(), \
        frame.iloc[np.flatnonzero(valid)].copy(), y.iloc[np.flatnonzero(valid)].copy(), categories


def fit_catboost_regression(x_train, y_train, x_valid, y_valid, categorical_indices, parameters, early_stopping_rounds):
    """Fit one explicit CPU model with one native thread and no backend file writes."""
    from catboost import CatBoostRegressor
    if not isinstance(parameters, dict) or parameters.get('task_type', 'CPU') != 'CPU':
        raise ValueError('Explicit CPU training parameters required')
    forbidden = {'thread_count','allow_writing_files','train_dir','devices','device_config'}
    if forbidden & set(parameters):
        raise ValueError('Worker, device and file policies are controlled by the executor')
    if type(early_stopping_rounds) is not int or early_stopping_rounds < 1:
        raise ValueError('Positive early stopping patience required')
    if not isinstance(x_train,pd.DataFrame) or not isinstance(x_valid,pd.DataFrame) or list(x_train.columns) != list(x_valid.columns):
        raise ValueError('Identical ordered training and validation feature schemas required')
    if not x_train.columns.is_unique or not len(x_train.columns) or not len(x_train) or not len(x_valid):
        raise ValueError('Nonempty unique training schema and populations required')
    if not isinstance(categorical_indices,list) or any(type(i) is not int or not 0 <= i < len(x_train.columns) for i in categorical_indices) or len(set(categorical_indices)) != len(categorical_indices):
        raise ValueError('Unique in-range categorical indices required')
    for x, y in [(x_train,y_train),(x_valid,y_valid)]:
        values = np.asarray(y,dtype=np.float64)
        if values.shape != (len(x),) or not np.isfinite(values).all():
            raise ValueError('Finite aligned regression targets required')
        if isinstance(y,pd.Series) and not y.index.equals(x.index):
            raise ValueError('Target row index differs from feature rows')
    model = CatBoostRegressor(**dict(parameters,thread_count=1,allow_writing_files=False))
    model.fit(x_train,y_train,eval_set=(x_valid,y_valid),cat_features=categorical_indices,
        use_best_model=True,verbose=False,early_stopping_rounds=early_stopping_rounds)
    return model
