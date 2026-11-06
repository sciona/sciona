"""Publication must reject incomplete, weakened or differently typed evidence."""
from copy import deepcopy
import json

import pytest

from scripts import future_sales_publication_evidence as evidence


def reports():
    return [json.loads((evidence.ROOT / path).read_text())
            for path in (evidence.EXECUTION, evidence.SOURCE)]


def test_exact_reports():
    execution, source = reports()
    evidence.validate_reports(execution, source, deepcopy(execution), deepcopy(source))


@pytest.mark.parametrize('section', ['code_sha256', 'provider_sha256', 'cases', 'resource_policy', 'limits'])
def test_missing_coverage(section):
    execution, source = reports()
    changed = deepcopy(execution)
    block=changed['execution'][section]
    if isinstance(block,dict):block.pop(next(iter(block)))
    else:block.pop()
    with pytest.raises(ValueError):
        evidence.validate_reports(changed, source, execution, source)


@pytest.mark.parametrize('field,value', [
    ('passed', 1), ('approved', 0), ('catalog_mutations', False),
    ('nodes', 9.0), ('nodes', 8), ('edges', 29), ('distinct_providers', 8),
    ('stored_catalog_graph_executed', False), ('synthetic_only', False),
    ('version_id', 'wrong'), ('artifact_id', 'wrong'),
    ('graph_sha256', '0' * 64), ('cases', []), ('limits', []),
])
def test_execution_changes(field, value):
    execution, source = reports()
    changed = deepcopy(execution)
    target=changed['execution'] if field in ('nodes','edges','distinct_providers','cases') else changed
    target[field] = value
    with pytest.raises(ValueError):
        evidence.validate_reports(changed, source, execution, source)


@pytest.mark.parametrize('case', range(3))
@pytest.mark.parametrize('field,value', [
    ('all_uncached', 1), ('registered_rolling_identity', False),
    ('full_result_equal', False), ('forecast_rows', 2),
    ('trials', 0), ('rolling_calls', 0),
])
def test_incomplete_case(case, field, value):
    execution, source = reports()
    changed = deepcopy(execution)
    changed['execution']['cases'][case][field] = value
    with pytest.raises(ValueError):
        evidence.validate_reports(changed, source, execution, source)


@pytest.mark.parametrize('field,value', [
    ('passed', 1), ('proposed_tier', 1), ('providers', []),
    ('software_sha256', {}), ('limitations', []),
    ('tests', {}), ('existing_providers', []),
])
def test_source_changes(field, value):
    execution, source = reports()
    changed = deepcopy(source)
    changed[field] = value
    with pytest.raises(ValueError):
        evidence.validate_reports(execution, changed, execution, source)


def test_changed_committed_report(tmp_path, monkeypatch):
    path = tmp_path / evidence.EXECUTION
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    original = (evidence.ROOT / evidence.EXECUTION).read_bytes()
    monkeypatch.setattr(evidence, 'ROOT', tmp_path)
    monkeypatch.setattr(evidence.subprocess, 'check_output', lambda *a, **k: original)
    with pytest.raises(ValueError, match='Immutable qualification changed'):
        evidence.committed_report(evidence.EXECUTION)
