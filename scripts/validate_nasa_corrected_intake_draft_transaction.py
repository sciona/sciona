"""Verify rollback of the new version while preserving the existing original artifact."""
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import scripts.import_nasa_corrected_intake as importer
from scripts.plan_nasa_corrected_intake import ROOT,plan,origin_rows,check_source_anchor,require,sha


def unchanged(proposed):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_source_anchor(db)
        require(origin_rows(db)==proposed['original_rows_sha256'],'Original history changed')
        require(not db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s',(proposed['version_id'],)).fetchone(),'Candidate version survived rollback')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE runner_version=%s',(importer.RUNNER,)).fetchone(),'Candidate audit survived rollback')
        row=db.execute('SELECT a.status,a.is_publishable,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(proposed['original_version_id'],)).fetchone()
        require(row==dict(status='draft',is_publishable=False,is_latest=True),'Original selection state changed')


def validate():
    proposed=plan();unchanged(proposed)
    rollback=importer.stage(False);unchanged(proposed)
    original=importer.ensure_row;writes=0
    def fault(*args,**kwargs):
        nonlocal writes
        result=original(*args,**kwargs);writes+=int(result)
        if writes==3:raise RuntimeError('injected corrected-intake failure')
        return result
    with patch.object(importer,'ensure_row',side_effect=fault):
        try:importer.stage(False)
        except RuntimeError as error:require(str(error)=='injected corrected-intake failure','Unexpected import error')
        else:raise ValueError('Injected failure did not fire')
    require(writes==3,'Failure must follow actual writes');unchanged(proposed)
    return dict(passed=True,rollback=rollback,failure_rollback_verified=True,injected_after_actual_writes=writes,
        committed_catalog_mutations=0,importer_sha256=sha(Path(importer.__file__)),validator_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/nasa_corrected_intake_draft_transaction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
