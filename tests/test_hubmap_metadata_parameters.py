"""Synthetic metadata checks for multi-input, multi-output recipe adapters."""
import pytest
from scripts import reconcile_hubmap_metadata as metadata


def test_optional_position_and_all_outputs_preserved():
    port=dict(direction='input',name='geometry',type_desc='object',ordinal=3,
              required=False,default_value_repr='None',constraints='Synthetic optional geometry.')
    output=lambda name,position:dict(direction='output',name=name,type_desc='object',ordinal=position,
        required=True,default_value_repr='',constraints='Synthetic '+name)
    contract=dict(plain_description='Use the supplied geometry.',ports=[output('second',1),port,output('first',0)])
    row=metadata.parameter_row(port,contract,dict(limitations=['Synthetic limitation'],reuse_decision='Explicit runtime inputs'))
    assert (row['required'],row['default_value_repr'],row['position'])==(False,'None',3)
    assert [p['name'] for p in row['constraints_json']['output_contracts']]==['first','second']
    assert row['constraints_json']['validity_bounds']==['Synthetic limitation']
    port=dict(port,required=True,default_value_repr='',ordinal=0)
    row=metadata.parameter_row(port,contract,dict(limitations=['Synthetic limitation'],reuse_decision='Explicit runtime inputs'))
    assert (row['required'],row['default_value_repr'],row['position'])==(True,'',0)


def test_missing_output_rejected():
    with pytest.raises(ValueError):metadata.parameter_row(dict(direction='input'),dict(ports=[]),{})


def test_output_not_parameter():
    with pytest.raises(ValueError):metadata.parameter_row(dict(direction='output'),{}, {})


def test_unqualified_metadata_rejected(monkeypatch):
    monkeypatch.setattr(metadata,'COMMIT',None)
    with pytest.raises(ValueError,match='Committed'):metadata.qualified()
