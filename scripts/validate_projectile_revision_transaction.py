"""Verify rollback and injected write failure for the original physics revision."""
import argparse
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.stage_projectile_original_revision import ROOT,ARTIFACT,ORIGINAL,RUNNER,history,stage,sha,require


def snapshot():
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
        require(state==dict(status='draft',is_publishable=False),'Original approval changed')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE artifact_id=%s AND runner_version=%s',(ARTIFACT,RUNNER)).fetchone(),'Staging audit survived rollback')
        versions=db.execute('SELECT version_id::text,content_hash,is_latest,trust_tier FROM artifact_versions WHERE artifact_id=%s ORDER BY version_id',(ARTIFACT,)).fetchall()
        require([r['version_id'] for r in versions if r['is_latest']]==[ORIGINAL],'Original latest selection changed')
        return dict(versions=versions,history=history(db))


def validate(source):
    before=snapshot();rollback=stage(source)
    require(snapshot()==before,'Dry-run rollback changed catalog')
    try:stage(source,fail_after_writes=3)
    except RuntimeError as error:require(str(error)=='injected revision staging failure','Unexpected injected failure')
    else:raise ValueError('Write failure did not fire')
    require(snapshot()==before,'Injected write rollback changed catalog')
    return dict(passed=True,rollback_verified=True,injected_failure_rollback_verified=True,
        injected_after_actual_writes=3,committed_catalog_mutations=0,draft=rollback,
        stager_sha256=sha(ROOT/'scripts/stage_projectile_original_revision.py'),validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/projectile_original_revision_transaction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rollback_verified=True,injected_failure_rollback_verified=True,committed_catalog_mutations=0)))
