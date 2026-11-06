"""Publication must reject incomplete, weakened or differently typed evidence."""
from copy import deepcopy
import json

import pytest

from scripts import march_publication_evidence as evidence


def reports():
    return [json.loads((evidence.ROOT / path).read_text())
            for path in (evidence.EXECUTION, evidence.SOURCE)]


def test_exact_reports():
    execution, source = reports()
    evidence.validate_reports(execution, source, deepcopy(execution), deepcopy(source))


@pytest.mark.parametrize('section', ['code_sha256', 'provider_sha256', 'evidence_sha256',
                                    'runtime', 'resource_policy'])
def test_missing_coverage(section):
    execution, source = reports()
    changed = deepcopy(execution)
    changed[section].pop(next(iter(changed[section])))
    with pytest.raises(ValueError):
        evidence.validate_reports(changed, source, execution, source)


@pytest.mark.parametrize('field,value', [
    ('passed', 1), ('approved', 0), ('catalog_mutations', False),
    ('nodes', 10.0), ('nodes', 9), ('edges', 15), ('distinct_providers', 8),
    ('stored_catalog_graph_executed', False), ('synthetic_only', False),
    ('version_id', 'wrong'), ('artifact_id', 'wrong'),
    ('serialized_graph_sha256', '0' * 64), ('cases', []), ('limits', []),
])
def test_execution_changes(field, value):
    execution, source = reports()
    changed = deepcopy(execution)
    changed[field] = value
    with pytest.raises(ValueError):
        evidence.validate_reports(changed, source, execution, source)


@pytest.mark.parametrize('case', range(3))
@pytest.mark.parametrize('field,value', [
    ('all_uncached', 1), ('probability_bytes_equal', False),
    ('full_result_equal', False), ('predicted_matchups', 2),
    ('all_ordered_pairs', False), ('orientation_symmetry', False),
])
def test_incomplete_case(case, field, value):
    execution, source = reports()
    changed = deepcopy(execution)
    changed['cases'][case][field] = value
    with pytest.raises(ValueError):
        evidence.validate_reports(changed, source, execution, source)


@pytest.mark.parametrize('field,value', [
    ('passed', 1), ('proposed_tier', 1), ('original_stages', 4),
    ('source_stage_mapping', {}), ('limitations', []),
    ('complete_stored_cases', 2), ('candidate_version_id', 'wrong'),
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
