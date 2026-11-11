from types import SimpleNamespace
import pytest
from scripts.review_causal_decomposed_bindings import check_call


def operation(values,enabled=True):return values


def fixture():
    node=SimpleNamespace(node_id='synthetic',matched_primitive='synthetic.operation',
        inputs=[SimpleNamespace(name='values',type_desc='numpy.ndarray')],
        outputs=[SimpleNamespace(name='result',type_desc='numpy.ndarray')])
    ports=[dict(direction='input',name='values',type_desc='numpy.ndarray',required=True,default_value_repr='',ordinal=0),
        dict(direction='input',name='enabled',type_desc='bool',required=False,default_value_repr='True',ordinal=1),
        dict(direction='output',name='result',type_desc='numpy.ndarray',required=True,default_value_repr='',ordinal=0)]
    return node,ports


def test_native_default_is_recorded_when_omitted():
    node,ports=fixture()
    assert check_call(node,operation,ports)['omitted_native_defaults']=={'enabled':'True'}


@pytest.mark.parametrize('mutation',['default','required','output_alias','output_ordinal','input_type'])
def test_corrupt_catalog_contract_is_rejected(mutation):
    node,ports=fixture()
    if mutation=='default':ports[1]['default_value_repr']='False'
    elif mutation=='required':ports[0]['required']=False
    elif mutation=='output_alias':ports[2]['name']='unexpected'
    elif mutation=='output_ordinal':ports[2]['ordinal']=1
    else:ports[0]['type_desc']='list'
    with pytest.raises(ValueError):check_call(node,operation,ports)
