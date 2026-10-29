"""Semantic review must reject stale or incomplete numerical evidence."""
import hashlib
import json

import pytest

from scripts import review_flavours_provider_semantics as reviewer


def fixture_report(tmp_path,monkeypatch):
    monkeypatch.setattr(reviewer,'ROOT',tmp_path)
    directory=tmp_path/'docs/reviews'
    directory.mkdir(parents=True)
    source=tmp_path/'synthetic_operation.py'
    source.write_text('def identity(value): return value\n')
    report=dict(passed=True,synthetic_only=True,test_results=dict(tests=3,failures=0,errors=0,skipped=0),
                source_sha256={'synthetic_operation.py':hashlib.sha256(source.read_bytes()).hexdigest()})
    return directory/'synthetic.json',source,report


def test_current_complete_evidence_is_accepted(tmp_path,monkeypatch):
    path,_,report=fixture_report(tmp_path,monkeypatch)
    path.write_text(json.dumps(report))
    actual,digest=reviewer.verify_report('synthetic')
    assert actual==report and digest==hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('fault',['failed','not_synthetic','zero_tests','skipped','error','source_drift','missing_pins','runtime_drift'])
def test_incomplete_or_stale_evidence_is_rejected(tmp_path,monkeypatch,fault):
    path,source,report=fixture_report(tmp_path,monkeypatch)
    if fault=='failed':report['passed']=False
    elif fault=='not_synthetic':report['synthetic_only']=False
    elif fault=='zero_tests':report['test_results']['tests']=0
    elif fault=='skipped':report['test_results']['skipped']=1
    elif fault=='error':report['test_results']['errors']=1
    elif fault=='source_drift':source.write_text('def identity(value): return None\n')
    elif fault=='missing_pins':report['source_sha256']={}
    else:
        report['runtime_versions']={'numpy':'qualified-version'}
        monkeypatch.setattr(reviewer.metadata,'version',lambda name:'different-version')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):reviewer.verify_report('synthetic')


def test_every_provider_has_one_explicit_reuse_scope():
    records=json.loads((reviewer.ROOT/'docs/reviews/flavours_provider_post_reconciliation_inventory.json').read_text())['providers']
    names=[n for _,members,*_ in reviewer.GROUPS for n in members]
    assert len(names)==len(set(names))==39
    assert set(names)=={r['runtime_fqdn'].rsplit('.',1)[1] for r in records}
    groups={name:(reuse,bounds) for _,members,reuse,_,bounds,_ in reviewer.GROUPS for name in members}
    assert groups['fit_configured_binary_model'][0]=='generic'
    assert 'NaN' in groups['fit_configured_binary_model'][1]
    assert groups['mass_regression_inputs'][0]=='recipe_adapter'
    assert 'state=None' in groups['standard_scaler_partial_fit'][1]
    assert 'static propagation' in groups['discrimination_inputs'][1]
