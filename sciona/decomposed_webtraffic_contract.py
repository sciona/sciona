"""Explicit complete forecasting boundary; no hidden feature preparation or fit."""
from dataclasses import fields
import datetime
import json
import numpy as np
import pandas as pd
from .webtraffic_runtime import RuntimeConfig


def unpack(payload):
    # A finite JSON copy prevents caller-owned lists/configuration changing while
    # model training executes. Unknown targets remain explicit null counts.
    try:payload=json.loads(json.dumps(payload,allow_nan=False))
    except (TypeError,ValueError,OverflowError) as error:raise ValueError('Finite JSON payload required') from error
    if not isinstance(payload,dict) or set(payload)!={'version','pages','start_day','counts','config','output'} or type(payload['version']) is not int or payload['version']!=2:
        raise ValueError('Version2 complete forecasting payload required')
    pages=payload['pages'];rows=payload['counts'];options=payload['config']
    if (not isinstance(pages,list) or not pages or any(not isinstance(p,str) or not p for p in pages)
        or len(set(pages))!=len(pages) or pages!=sorted(pages)):
        raise ValueError('Unique sorted runtime page strings required')
    if not isinstance(rows,list) or len(rows)!=len(pages) or any(not isinstance(r,list) for r in rows):
        raise ValueError('One numeric row per runtime page required')
    days=len(rows[0])
    if days==0 or any(len(row)!=days for row in rows):raise ValueError('Rectangular daily count rows required')
    if any(v is not None and (type(v) not in (int,float) or not np.isfinite(v) or v<0) for row in rows for v in row):
        raise ValueError('Nonnegative finite counts or null required')
    if not isinstance(options,dict) or set(options)-{f.name for f in fields(RuntimeConfig)}:
        raise ValueError('Unknown runtime configuration field')
    day=payload['start_day']
    if not isinstance(day,str) or datetime.date.fromisoformat(day).isoformat()!=day:
        raise ValueError('ISO calendar start day required')
    with np.errstate(over='ignore',invalid='ignore'):values=np.array(rows,dtype=np.float32)
    if np.isinf(values).any():raise ValueError('Count magnitude exceeds float32 representation')
    frame=pd.DataFrame(values,index=np.array(pages,dtype=object),columns=pd.date_range(day,periods=days))
    config=RuntimeConfig(**options);config.validate()
    output=payload['output']
    if not isinstance(output,dict) or set(output)!={'rows','columns','identifiers'}:
        raise ValueError('Explicit aligned output coordinates and identifiers required')
    out_rows=output['rows'];out_columns=output['columns'];identifiers=output['identifiers']
    if out_rows!=pages:raise ValueError('Output row order must match the complete runtime population')
    forecast_days=[d.date().isoformat() for d in pd.date_range(frame.columns[-1]+pd.Timedelta(days=1),periods=63)]
    if (not isinstance(out_columns,list) or not out_columns or len(out_columns)>63
        or forecast_days[-len(out_columns):]!=out_columns):
        raise ValueError('Output columns must be an exact suffix of the complete forecast horizon')
    if (not isinstance(identifiers,list) or len(identifiers)!=len(pages)
        or any(not isinstance(row,list) or len(row)!=len(out_columns) for row in identifiers)
        or any(type(value) is not int or not -(2**63)<=value<2**63 for row in identifiers for value in row)):
        raise ValueError('Aligned int64 output identifiers required')
    return (frame,config,0.,63,0,[365,91],[1.5,2.],out_rows,out_columns,np.array(identifiers,dtype=np.int64))


def forecast_axes(frame):
    """Expose numeric panel and coordinate axes for generic output alignment."""
    if not isinstance(frame,pd.DataFrame) or frame.empty or not isinstance(frame.columns,pd.DatetimeIndex):
        raise ValueError('Dated finalized prediction panel required')
    if frame.to_numpy().dtype!=np.dtype('int64'):raise ValueError('Finalized int64 predictions required')
    return frame.to_numpy(copy=True),frame.index.tolist(),[day.date().isoformat() for day in frame.columns]


def result(identifiers,values):
    """Return finite JSON record columns without dataset-specific output names."""
    if (not isinstance(identifiers,np.ndarray) or not isinstance(values,np.ndarray)
        or identifiers.dtype!=np.dtype('int64') or values.dtype!=np.dtype('int64')
        or identifiers.ndim!=1 or values.shape!=identifiers.shape or not len(values)):
        raise ValueError('Aligned nonempty int64 record columns required')
    return dict(version=2,identifiers=identifiers.tolist(),values=values.tolist())
