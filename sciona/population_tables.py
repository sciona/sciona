"""Combine explicitly selected population features with caller-provided labels."""
import pandas as pd


def concatenate_labeled_populations(tables, labels, columns, label_column, order_column):
    """Select each table's ordered columns, label, concatenate, and sort.

    Column selection and population order are explicit. Different selected
    columns produce a union with pandas missing-value semantics. Sorting uses
    pandas quicksort to preserve the qualified source behavior; ties do not
    carry a stable-order guarantee. No caller table is mutated.
    """
    if not isinstance(tables,list) or not tables or not len(tables)==len(labels)==len(columns):
        raise ValueError('Aligned nonempty population lists required')
    if not all(isinstance(v,str) and v for v in labels) or len(set(labels))!=len(labels):
        raise ValueError('Unique nonempty population labels required')
    if not isinstance(label_column,str) or not label_column or label_column==order_column:
        raise ValueError('Distinct label and ordering columns required')
    combined=pd.DataFrame()
    for table,label,selected in zip(tables,labels,columns):
        if not isinstance(table,pd.DataFrame) or not table.columns.is_unique:
            raise ValueError('DataFrames with unique columns required')
        if not isinstance(selected,list) or not selected or len(set(selected))!=len(selected):
            raise ValueError('Unique ordered column selection required')
        if order_column not in selected or label_column in table.columns or any(c not in table.columns for c in selected):
            raise ValueError('Missing ordering/selected column or existing label column')
        part=table.loc[:,selected].copy()
        part[label_column]=label
        combined=pd.concat([combined,part])
    return combined.sort_values(order_column,kind='quicksort').reset_index(drop=True)
