"""Domain-independent exact alignment of integer panel outputs."""
import numpy as np


def _labels(values):
    if not isinstance(values,list) or not values or any(type(v) is not str or not v for v in values):
        raise ValueError('Nonempty string axis labels required')
    if len(set(values))!=len(values):raise ValueError('Axis labels must be unique')


def aligned_suffix_records(values: np.ndarray,row_labels: list,column_labels: list,
                           output_rows: list,output_columns: list,identifiers: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Select an exact column suffix and flatten aligned IDs/values in row order.

    Inputs are already finalized int64 predictions. No sorting, rounding, missing
    fills, date parsing or domain-specific naming occurs here. IDs may repeat;
    callers retain responsibility for their output identity policy.
    """
    for labels in (row_labels,column_labels,output_rows,output_columns):_labels(labels)
    if not isinstance(values,np.ndarray) or values.dtype!=np.dtype('int64') or values.shape!=(len(row_labels),len(column_labels)):
        raise ValueError('Aligned rank-two int64 values required')
    if not isinstance(identifiers,np.ndarray) or identifiers.dtype!=np.dtype('int64') or identifiers.shape!=(len(output_rows),len(output_columns)):
        raise ValueError('Aligned rank-two int64 identifiers required')
    if output_rows!=row_labels:raise ValueError('Output rows must exactly match input order')
    if len(output_columns)>len(column_labels) or column_labels[-len(output_columns):]!=output_columns:
        raise ValueError('Output columns must be an exact nonempty suffix')
    selected=values[:,-len(output_columns):]
    return identifiers.flatten(order='C'),selected.flatten(order='C')
