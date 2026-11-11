"""Synthetic checks that malformed gate records cannot open a transaction.

Training review is stubbed here. These tests do not qualify real publication.
"""
import json
from pathlib import Path

import pytest
from scripts import promote_aptos_execution as publisher


class ReachedTransaction(Exception):
    pass


@pytest.fixture
def gate_record(tmp_path, monkeypatch):
    (tmp_path / 'docs/reviews').mkdir(parents=True)
    script = tmp_path / 'scripts/validate_aptos_publication_gates.py'
    script.parent.mkdir()
    script.write_text('# Synthetic validator identity\n')
    monkeypatch.setattr(publisher, 'review', lambda root: (None,) * 8 + ({},))
    def transaction(*args, **kwargs):
        raise ReachedTransaction('Synthetic transaction boundary; no database connection')
    monkeypatch.setattr(publisher.psycopg, 'connect', transaction)
    # dotenv lookup precedes connect; supply only a synthetic connection value.
    monkeypatch.setattr(publisher, 'dotenv_values', lambda path: {'SCIONA_DATA_CATALOG_DATABASE_URL': 'synthetic'})
    record = dict(format='aptos-publication-gates.v1', result='passed',
        rejected_gates=sorted(publisher.EXPECTED_GATES),
        publisher_sha256=publisher.sha(Path(publisher.__file__)),
        validator_sha256=publisher.sha(script))
    return tmp_path, record


@pytest.mark.parametrize('fault', ['format', 'result', 'missing_case', 'duplicate_case',
                                  'foreign_case', 'publisher_hash', 'validator_hash'])
def test_invalid_gate_record_stops_before_transaction(gate_record, fault):
    root, record = gate_record
    if fault in ['format', 'result']:
        record[fault] = 'invalid'
    elif fault == 'missing_case':
        record['rejected_gates'].pop()
    elif fault == 'duplicate_case':
        record['rejected_gates'][-1] = record['rejected_gates'][0]
    elif fault == 'foreign_case':
        record['rejected_gates'][-1] = 'unreviewed_case'
    else:
        record[fault.replace('_hash', '_sha256')] = 'invalid'
    (root / 'docs/reviews/competition_aptos_publication_gates.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Publication gate evidence missing or stale'):
        publisher.promote(root)


def test_exact_record_reaches_only_mock_transaction(gate_record):
    root, record = gate_record
    (root / 'docs/reviews/competition_aptos_publication_gates.json').write_text(json.dumps(record))
    with pytest.raises(ReachedTransaction):
        publisher.promote(root)
