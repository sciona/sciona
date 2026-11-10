"""An unqualified HuBMAP candidate must not reach database staging."""
import pytest
from scripts import stage_hubmap_corrected_intake as staging


@pytest.mark.parametrize('operation', ['sources', 'make_plan'])
def test_missing_committed_provider_review_prevents_database_access(monkeypatch, operation):
    monkeypatch.setattr(staging.metadata, 'COMMIT', None)

    def forbidden(*args, **kwargs):
        pytest.fail('Unqualified staging reached a database or graph builder')

    monkeypatch.setattr(staging, 'connect', forbidden)
    monkeypatch.setattr(staging, 'build_catalog_hubmap_graph', forbidden)
    with pytest.raises(ValueError, match='Committed complete-graph provider contract review required'):
        getattr(staging, operation)()
