"""Synthetic tests for exact suffix output alignment across arbitrary domains."""
import numpy as np
import pytest
from sciona.panel_output import aligned_suffix_records


@pytest.mark.parametrize('rows,width,suffix',[(1,1,1),(2,5,2),(7,9,9),(4,8,1)])
@pytest.mark.parametrize('strided',[False,True])
def test_exact_values_order_and_no_alias(rows,width,suffix,strided):
    a=np.arange(rows*width*(2 if strided else 1),dtype=np.int64).reshape(rows,-1)-50
    values=a[:,::2] if strided else a
    ids=np.arange(rows*suffix,dtype=np.int64).reshape(rows,suffix)+100
    r=[f'entity-{i}' for i in range(rows)];c=[f'position-{i}' for i in range(width)]
    saved=values.copy();idcopy=ids.copy()
    actual_ids,actual=aligned_suffix_records(values,r,c,r.copy(),c[-suffix:],ids)
    assert actual.tolist()==[int(values[i,j]) for i in range(rows) for j in range(width-suffix,width)]
    assert actual_ids.tolist()==[int(ids[i,j]) for i in range(rows) for j in range(suffix)]
    assert not np.shares_memory(actual,values) and not np.shares_memory(actual_ids,ids)
    np.testing.assert_array_equal(values,saved);np.testing.assert_array_equal(ids,idcopy)


@pytest.mark.parametrize('fault',[
    'row_order','column_order','nonsuffix','empty','duplicate_rows','duplicate_columns',
    'float_values','float_ids','wrong_values_shape','wrong_ids_shape','nonstring','tuple_labels'])
def test_invalid_alignment_rejected(fault):
    v=np.arange(6,dtype=np.int64).reshape(2,3);ids=np.ones((2,2),dtype=np.int64)
    r=['a','b'];c=['x','y','z'];out_r=r.copy();out_c=c[1:]
    if fault=='row_order':out_r.reverse()
    elif fault=='column_order':out_c.reverse()
    elif fault=='nonsuffix':out_c=c[:2]
    elif fault=='empty':out_c=[]
    elif fault=='duplicate_rows':r=['a','a']
    elif fault=='duplicate_columns':c=['x','x','z']
    elif fault=='float_values':v=v.astype(float)
    elif fault=='float_ids':ids=ids.astype(float)
    elif fault=='wrong_values_shape':v=v.ravel()
    elif fault=='wrong_ids_shape':ids=ids.ravel()
    elif fault=='nonstring':r=[1,2]
    else:r=tuple(r)
    with pytest.raises(ValueError):aligned_suffix_records(v,r,c,out_r,out_c,ids)


def test_int64_extremes_and_repeated_identifiers_preserved():
    values=np.array([[np.iinfo(np.int64).min,np.iinfo(np.int64).max]],dtype=np.int64)
    ids=np.array([[7,7]],dtype=np.int64)
    a,b=aligned_suffix_records(values,['signal'],['low','high'],['signal'],['low','high'],ids)
    assert a.tolist()==[7,7] and b.tolist()==values[0].tolist()
