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


@pytest.mark.parametrize('failed_catalog',['artifact','atom'])
def test_failed_evidence_in_either_catalog_rejected(failed_catalog):
    identity=dict(artifact_id='synthetic-id',artifact_kind='atom',fqdn='synthetic.provider',
        status='approved',is_publishable=True,version_id='synthetic-version',content_hash='a'*64,trust_tier=3)
    wanted={k:identity[k] for k in ('fqdn','status','is_publishable','version_id','content_hash','trust_tier')}
    class Result:
        def __init__(self,rows):self.rows=rows
        def fetchall(self):return self.rows
        def fetchone(self):return self.rows[0] if self.rows else None
    class Database:
        def execute(self,sql,args):
            if 'JOIN' in sql:return Result([wanted])
            if '_audit_evidence' in sql:return Result([{'failed':True}] if failed_catalog+'_audit_evidence' in sql else [])
            raise AssertionError('Unexpected query before failed audit rejection')
    with pytest.raises(ValueError,match='Failed retained audit in '+failed_catalog):
        metadata.check_identity(Database(),dict(identity=identity,catalogs=dict(artifact={},atom={})))
