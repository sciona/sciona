"""Publication cannot touch the database without immutable execution evidence."""
import pytest
from scripts import promote_hubmap_corrected_intake as publisher


@pytest.mark.parametrize('commit',[None,'','a'*8,'g'*40,True])
@pytest.mark.parametrize('apply',[False,True])
def test_missing_execution_evidence_prevents_publication(monkeypatch,commit,apply):
    monkeypatch.setattr(publisher.evidence,'EVIDENCE_COMMIT',commit)

    def forbidden(*args,**kwargs):
        pytest.fail('Unqualified publication attempted database or source access')

    monkeypatch.setattr(publisher.intake,'connect',forbidden)
    monkeypatch.setattr(publisher.subprocess,'check_output',forbidden)
    with pytest.raises(ValueError,match='Immutable published execution evidence commit required'):
        publisher.publish(apply=apply)
