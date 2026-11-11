"""Stored execution must not start from uncommitted or abbreviated staging evidence."""
import json
import pytest
from scripts import validate_hubmap_corrected_execution as execution


@pytest.mark.parametrize('commit',[None,'','a'*8,'g'*40,True])
def test_uncommitted_staging_rejected_before_graph_or_database_access(monkeypatch,commit):
    monkeypatch.setattr(execution,'STAGING_COMMIT',commit)

    def forbidden(*args,**kwargs):
        pytest.fail('Invalid staging identity reached qualified state access')

    monkeypatch.setattr(execution.staging,'qualified_plan',forbidden)
    monkeypatch.setattr(execution.subprocess,'check_output',forbidden)
    with pytest.raises(ValueError,match='Full committed staging evidence required'):
        execution.qualified_staging()


@pytest.mark.parametrize('fault',[None,'failed_gates','stale_builder','stale_plan','wrong_version',
    'wrong_graph','already_approved','not_applied','repeat_writes','boolean_writes','lost_history','changed_file'])
def test_staging_evidence_guards(tmp_path,monkeypatch,fault):
    plan=dict(passed=True,version_id='synthetic-version',graph_sha256='b'*64)
    applied=dict(applied=True,approved=False,writes=0,source_histories_preserved=True,
                 version_id=plan['version_id'],graph_sha256=plan['graph_sha256'])
    records=dict(plan=plan,gates=dict(passed=True,builder_sha256='a'*64,plan_sha256='a'*64),
                 apply=dict(applied),repeat=dict(applied))
    if fault=='failed_gates':records['gates']['passed']=False
    elif fault=='stale_builder':records['gates']['builder_sha256']='c'*64
    elif fault=='stale_plan':records['gates']['plan_sha256']='c'*64
    elif fault=='wrong_version':records['apply']['version_id']='other'
    elif fault=='wrong_graph':records['repeat']['graph_sha256']='c'*64
    elif fault=='already_approved':records['apply']['approved']=True
    elif fault=='not_applied':records['apply']['applied']=False
    elif fault=='repeat_writes':records['repeat']['writes']=1
    elif fault=='boolean_writes':records['repeat']['writes']=False
    elif fault=='lost_history':records['apply']['source_histories_preserved']=False
    raw={}
    for name,record in records.items():
        path='docs/reviews/hubmap_corrected_intake_'+name+'.json'
        raw[path]=json.dumps(record).encode()
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw[path])
    if fault=='changed_file':(tmp_path/'docs/reviews/hubmap_corrected_intake_apply.json').write_text('{}')
    monkeypatch.setattr(execution,'ROOT',tmp_path)
    monkeypatch.setattr(execution,'STAGING_COMMIT','d'*40)
    monkeypatch.setattr(execution.subprocess,'check_output',lambda args,**kwargs:raw[args[2].split(':',1)[1]])
    monkeypatch.setattr(execution,'sha',lambda path:'a'*64)
    monkeypatch.setattr(execution.staging,'qualified_plan',lambda:plan)
    if fault is None:
        assert execution.qualified_staging()==plan
    else:
        with pytest.raises(ValueError):execution.qualified_staging()
