import pandas as pd
import pytest
from sciona.population_tables import concatenate_labeled_populations as combine


def test_union_alignment_order_and_nonmutation():
    first=pd.DataFrame({'order':[3,1],'value':[30,10],'ignored':[9,9]})
    second=pd.DataFrame({'order':[2],'extra':[20]})
    original=first.copy(deep=True)
    result=combine([first,second],['manufacturing','energy'],[['order','value'],['order','extra']],'population','order')
    assert result['order'].tolist()==[1,2,3]
    assert result['population'].tolist()==['manufacturing','energy','manufacturing']
    assert pd.isna(result.loc[1,'value']) and pd.isna(result.loc[0,'extra'])
    pd.testing.assert_frame_equal(first,original)
    assert 'ignored' not in result


@pytest.mark.parametrize('labels,columns,label_column',[
    (['same','same'],[['order'],['order']],'population'),
    (['a'],[['order'],['order']],'population'),
    (['a','b'],[['missing'],['order']],'population'),
    (['a','b'],[['order','order'],['order']],'population'),
    (['a','b'],[['order'],['order']],'order'),
])
def test_invalid_population_contract(labels,columns,label_column):
    with pytest.raises(ValueError):combine([pd.DataFrame({'order':[1]}),pd.DataFrame({'order':[2]})],labels,columns,label_column,'order')
