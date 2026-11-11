"""A served-state check must not claim success without stored execution evidence."""
import pytest
from scripts import verify_hubmap_corrected_served as served


def test_unqualified_serving_check_stops_before_database_access(monkeypatch):
    monkeypatch.setattr(served.publisher.evidence,'EVIDENCE_COMMIT',None)

    def forbidden(*args,**kwargs):
        pytest.fail('Unqualified served check reached the database')

    monkeypatch.setattr(served.intake,'connect',forbidden)
    with pytest.raises(ValueError,match='Full immutable stored execution commit required'):
        served.verify()
