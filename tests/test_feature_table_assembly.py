import numpy as np
import pandas as pd
import pytest
from sciona.feature_table_assembly import assemble_feature_tables


def test_multiple_keys_query_order_duplicates_index_and_missing_matches():
    queries=pd.DataFrame({'entity':['b','a','b'],'tick':[2,1,2]},index=[9,3,9])
    by_entity=pd.DataFrame({'entity':['a','b'],'category':['A','B']})
    by_time=pd.DataFrame({'tick':[1],'value':[7.]})
    before=queries.copy(deep=True)
    result=assemble_feature_tables(queries,[(by_entity,['entity']),(by_time,['tick'])])
    pd.testing.assert_frame_equal(result[queries.columns],queries)
    pd.testing.assert_frame_equal(queries,before)
    assert result['category'].tolist()==['B','A','B']
    np.testing.assert_allclose(result['value'],[np.nan,7,np.nan],equal_nan=True)


@pytest.mark.parametrize('table,keys',[
    (pd.DataFrame({'key':[1,1],'value':[2,3]}),['key']),
    (pd.DataFrame({'key':[np.nan],'value':[2]}),['key']),
    (pd.DataFrame({'key':[1],'existing':[2]}),['key']),
    (pd.DataFrame({'key':[1],'value':[2]}),['missing']),
])
def test_ambiguous_joins_rejected(table,keys):
    with pytest.raises(ValueError):assemble_feature_tables(pd.DataFrame({'key':[1],'existing':[4]}),[(table,keys)])


def test_empty_queries_preserve_schema():
    queries=pd.DataFrame({'key':pd.Series(dtype=int)})
    result=assemble_feature_tables(queries,[(pd.DataFrame({'key':[1],'value':[2]}),['key'])])
    assert result.empty and result.columns.tolist()==['key','value']
