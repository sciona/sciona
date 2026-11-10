"""Source image inference routing over caller-decoded planes and raw metadata."""
import numpy as np
import pandas as pd

from sciona.osic_image_metadata import encode
from sciona.osic_image_preprocessing import inference_plane
from sciona.osic_predictions import image_trajectories


def select_slice_positions(names):
    """Keep strict source ordinal/count bounds, preserving directory order.

The denominator includes every supplied name, including excluded slices.
The source strips four trailing characters before parsing an integer. Caller
names and their mapping to decoded planes remain private runtime material.
"""
    if (not isinstance(names,(list,tuple)) or not names
            or any(not isinstance(n,str) or len(n)<=4 for n in names)
            or len(set(names))!=len(names)):
        raise ValueError('Ordered unique source-compatible slice names required')
    ordinals=[int(name[:-4]) for name in names]
    return np.array([i for i,n in enumerate(ordinals) if -.1 < n/len(names) < 1.1],dtype=np.int64)


def predict(baselines,queries,planes,predict_slices,*,decode_selected=None):
    """Prepare selected slices, predict slopes, and restore explicit query order.

``planes[entity]`` is an ordered list of (name, decoded integer 2D plane).
With an explicit ``decode_selected`` callable, payloads may instead be encoded;
only selected positions are decoded, preserving source handling of unused files.
The caller supplies a fitted predictor accepting [images,metadata] and returning
float32/64 scalar slopes of shape (slices,1). Its batching policy must preserve
the source inference batch size of 32. No fitting occurs in this adapter.
Baselines use one row per entity and int64/float64 numerical coordinates;
queries use explicit unique keys and integer times. Full coverage is required.
"""
    baseline_columns={'entity','time','target','numeric','category_a','category_b','uncertainty'}
    if decode_selected is not None and not callable(decode_selected):
        raise ValueError('A callable selected-plane decoder is required')
    for frame,columns in ((baselines,baseline_columns),(queries,{'key','entity','time'})):
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
        raise ValueError('Integer query times required by the source key policy')
    entities=baselines['entity'].tolist()
    if (not isinstance(planes,dict) or set(planes)!=set(entities)
            or not set(queries['entity']).issubset(entities) or not callable(predict_slices)):
        raise ValueError('Aligned image population and fitted predictor required')
    selected=[];slopes=[];prediction_dtype=None
    for entity in entities:
        records=planes[entity]
        if (not isinstance(records,(list,tuple)) or not records
                or any(not isinstance(r,(list,tuple)) or len(r)!=2 for r in records)):
            raise ValueError('Ordered named decoded planes required')
        positions=select_slice_positions([record[0] for record in records])
        if len(positions)<2:
            raise ValueError('Source cannot produce complete predictions with fewer than two selected slices')
        row=baselines.loc[baselines['entity']==entity].iloc[0]
        vector=encode(row['numeric'],row['category_a'],row['category_b'],branch='inference')
        pixels=[decode_selected(records[i][1]) if decode_selected is not None else records[i][1]
                for i in positions]
        images=np.expand_dims(np.stack([inference_plane(plane) for plane in pixels]),axis=-1)
        metadata=np.stack([vector for _ in positions])
        prediction=predict_slices([images,metadata])
        if (not isinstance(prediction,np.ndarray) or prediction.shape!=(len(positions),1)
                or prediction.dtype not in (np.dtype('float32'),np.dtype('float64'))
                or not np.isfinite(prediction).all()):
            raise ValueError('Finite aligned scalar slice predictions required')
        if prediction_dtype is not None and prediction.dtype!=prediction_dtype:
            raise ValueError('One fitted predictor must retain a consistent output dtype')
        prediction_dtype=prediction.dtype
        slopes.append(prediction)
        selected.append(positions)
    indices={entity:i for i,entity in enumerate(entities)}
    values=image_trajectories(slopes,baselines['time'].to_numpy(),baselines['target'].to_numpy(),
        baselines['uncertainty'].to_numpy(),np.array([indices[e] for e in queries['entity']],dtype=np.int64),
        queries['time'].to_numpy())
    return dict(keys=queries['key'].tolist(),values=values,selected_positions=selected)
