"""Prove rollback before and after latest-version activation of the original intake."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import scripts.promote_nasa_corrected_intake as publisher
from scripts.plan_nasa_corrected_intake import ROOT, plan, require, sha
from scripts.check_nasa_corrected_intake import check_candidate


def snapshot(proposed):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,proposed)
        result={}
        for table in ['artifacts','artifact_versions','artifact_descriptions','artifact_audit_rollups','artifact_audit_evidence']:
            rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE artifact_id=%s ORDER BY to_jsonb(t)::text',
                (proposed['artifact_id'],)).fetchall()
            result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s',
            (proposed['version_id'],publisher.RUNNER)).fetchone(),'Publication evidence survived rollback')
        return result


def validate():
    proposed=plan();baseline=snapshot(proposed)
    rollback=publisher.promote(False)
    require(snapshot(proposed)==baseline,'Dry-run rollback changed original artifact')
    original=publisher.ensure_row;writes=0
    def fail_after_write(*args,**kwargs):
        nonlocal writes
        result=original(*args,**kwargs);writes+=int(result)
        if writes==3:raise RuntimeError('injected publication write failure')
        return result
    with patch.object(publisher,'ensure_row',side_effect=fail_after_write):
        try:publisher.promote(False)
        except RuntimeError as error:require(str(error)=='injected publication write failure','Unexpected write failure')
        else:raise ValueError('Publication write failure did not fire')
    require(writes==3 and snapshot(proposed)==baseline,'Write failure rollback differs')
    original_check=publisher.check_candidate;activated=False
    def fail_after_activation(db,p,*,approved=False):
        nonlocal activated
        original_check(db,p,approved=approved)
        if approved:
            activated=True
            raise RuntimeError('injected post-activation failure')
    with patch.object(publisher,'check_candidate',side_effect=fail_after_activation):
        try:publisher.promote(False)
        except RuntimeError as error:require(str(error)=='injected post-activation failure','Unexpected activation failure')
        else:raise ValueError('Post-activation failure did not fire')
    require(activated and snapshot(proposed)==baseline,'Activation failure rollback differs')
    return dict(passed=True,approval_rollback=rollback,injected_transaction_failure_rolled_back=True,
        post_activation_failure_rolled_back=True,injected_after_actual_writes=writes,
        artifact_snapshot_sha256=baseline,committed_catalog_mutations=0,
        publisher_sha256=sha(Path(publisher.__file__)),validator_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/nasa_corrected_intake_publication_transaction_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,post_activation_failure_rolled_back=True,committed_catalog_mutations=0)))
