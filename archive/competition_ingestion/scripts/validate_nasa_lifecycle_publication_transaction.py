"""Verify approval rollback and injected mid-transaction failure recovery."""
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import scripts.promote_nasa_lifecycle as publisher
from scripts.review_conditional_correction import ROOT, require, sha
from scripts.plan_nasa_lifecycle_drafts import plan
from scripts.validate_nasa_lifecycle_database_gates import check_staged


def check_rollback(proposed):
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        check_staged(db, proposed)
        ids = [g['artifact_id'] for g in proposed['graphs'].values()] + [b['artifact_id'] for b in proposed['atoms']]
        require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=ANY(%s::uuid[])', (ids,)).fetchone(), 'Approval survived rollback')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE runner_version=%s', (publisher.RUNNER,)).fetchone(), 'Approval audit survived rollback')
        require(not db.execute("SELECT 1 FROM references_registry WHERE ref_id='nasa-lifecycle-drivendata-source'").fetchone(), 'Reference survived rollback')


def validate():
    proposed = plan()
    check_rollback(proposed)
    rollback = publisher.promote(False)
    check_rollback(proposed)
    original = publisher.ensure_row
    writes = 0
    def fail_after_write(*args, **kwargs):
        nonlocal writes
        result = original(*args, **kwargs)
        writes += int(result)
        if writes == 3:
            raise RuntimeError('injected publication transaction failure')
        return result
    with patch.object(publisher, 'ensure_row', fail_after_write):
        try:
            publisher.promote(False)
        except RuntimeError as error:
            require(str(error) == 'injected publication transaction failure', 'Unexpected transaction failure')
        else:
            raise ValueError('Transaction fault did not fire')
    require(writes == 3, 'Fault did not follow actual writes')
    check_rollback(proposed)
    return dict(passed=True, approval_rollback=rollback, injected_transaction_failure_rolled_back=True,
        injected_after_actual_writes=writes, publisher_sha256=sha(Path(publisher.__file__)),
        validator_sha256=sha(Path(__file__)), committed_catalog_mutations=0)


if __name__ == '__main__':
    report = validate()
    (ROOT / 'docs/reviews/nasa_lifecycle_publication_transaction_gates.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
