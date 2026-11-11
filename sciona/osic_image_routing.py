"""Sequential source population routing with explicitly injected computations."""
import numpy as np
import pandas as pd


def route_and_predict(baselines, queries, planes, model, batch_size,
                      prepare_function, select_function, predict_function):
    """Collect per-entity slice slopes without retaining prepared image populations.

    Preparation, input ordering and prediction are separate caller-supplied
    computations. This function validates population identity, preserves source
    order and returns explicit trajectory-reduction inputs; it does not reduce
    slopes or combine branches. Callables and their model code are trusted.
    """
    for frame,columns in ((baselines,{'entity','time','target','numeric','category_a','category_b','uncertainty'}),
                          (queries,{'key','entity','time'})):
        if (not isinstance(frame,pd.DataFrame) or frame.empty or not frame.columns.is_unique
                or set(frame.columns)!=columns):
            raise ValueError('Nonempty explicitly mapped baseline/query tables required')
        if not all(isinstance(v,str) and v for v in frame['entity']):
            raise ValueError('Nonempty entity identities required')
    if baselines['entity'].duplicated().any() or queries['key'].duplicated().any():
        raise ValueError('Unique baselines and query keys required')
    if not all(isinstance(v,str) and v for v in queries['key']):
        raise ValueError('Nonempty query keys required')
    for name in ('time','target','numeric','uncertainty'):
        values=baselines[name].to_numpy()
        if values.dtype not in (np.dtype('int64'),np.dtype('float64')) or not np.isfinite(values).all():
            raise ValueError('Finite int64/float64 baseline coordinates required')
    if queries['time'].dtype!=np.dtype('int64'):
        raise ValueError('Integer query times required by source key policy')
    entities=baselines['entity'].tolist()
    if (not isinstance(planes,dict) or set(planes)!=set(entities)
            or not set(queries['entity']).issubset(entities)
            or not all(callable(fn) for fn in (prepare_function,select_function,predict_function))):
        raise ValueError('Aligned image population and explicit callable computations required')
    if type(batch_size) is not int or batch_size!=32:
        raise ValueError('Explicit source batch size 32 required')
    slopes=[];selected=[];prediction_dtype=None
    for entity in entities:
        row=baselines.loc[baselines['entity']==entity].iloc[0]
        prepared=prepare_function(planes[entity],row['numeric'],row['category_a'],row['category_b'])
        positions=prepared['selected_positions']
        if (not isinstance(positions,np.ndarray) or positions.dtype!=np.int64 or positions.ndim!=1
                or len(positions)<2 or np.any(positions<0) or np.any(positions>=len(planes[entity]))
                or np.any(np.diff(positions)<=0)):
            raise ValueError('Ordered selected source positions required')
        inputs=select_function(prepared,['images','metadata'])
        prediction=predict_function(model,inputs,batch_size)
        if (not isinstance(prediction,np.ndarray) or prediction.shape!=(len(positions),1)
                or prediction.dtype not in (np.dtype('float32'),np.dtype('float64'))
                or not np.isfinite(prediction).all()):
            raise ValueError('Finite aligned scalar slice predictions required')
        if prediction_dtype is not None and prediction.dtype!=prediction_dtype:
            raise ValueError('One fitted predictor must retain a consistent output dtype')
        prediction_dtype=prediction.dtype
        # Retain scalar predictions and positions, not decoded images or model inputs.
        slopes.append(prediction);selected.append(positions)
        del prepared,inputs,prediction,positions
    indices={entity:i for i,entity in enumerate(entities)}
    return (slopes,baselines['time'].to_numpy(),baselines['target'].to_numpy(),
            baselines['uncertainty'].to_numpy(),
            np.array([indices[e] for e in queries['entity']],dtype=np.int64),
            queries['time'].to_numpy(),queries['key'].tolist(),selected)
