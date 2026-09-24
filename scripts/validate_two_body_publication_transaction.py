"""Verify publication rollback, including failure after latest-version activation."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import scripts.promote_two_body_revision as publisher
from scripts.check_two_body_revision import check
from scripts.stage_two_body_original_revision import ROOT,ARTIFACT,sha,require


def snapshot():
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check(db)
        result={}
        for table in ['artifacts','artifact_versions','artifact_descriptions','artifact_references','artifact_audit_rollups','artifact_audit_evidence','artifact_validity_bounds']:
            rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE artifact_id=%s ORDER BY to_jsonb(t)::text',(ARTIFACT,)).fetchall()
            result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        return result


def validate(source):
    baseline=snapshot();rollback=publisher.promote(source)
    require(snapshot()==baseline,'Publication dry run changed catalog')
    original=publisher.ensure_row;writes=0
    def fail_after_write(*args,**kwargs):
        nonlocal writes
        result=original(*args,**kwargs);writes+=int(result)
        if writes==3:raise RuntimeError('injected publication write failure')
        return result
    with patch.object(publisher,'ensure_row',side_effect=fail_after_write):
        try:publisher.promote(source)
        except RuntimeError as error:require(str(error)=='injected publication write failure','Unexpected publication error')
        else:raise ValueError('Write fault did not fire')
    require(writes==3 and snapshot()==baseline,'Write fault rollback differs')
    original_check=publisher.check_publication;activated=False
    def fail_after_activation(*args,**kwargs):
        nonlocal activated
        original_check(*args,**kwargs);activated=True
        raise RuntimeError('injected post-activation failure')
    with patch.object(publisher,'check_publication',side_effect=fail_after_activation):
        try:publisher.promote(source)
        except RuntimeError as error:require(str(error)=='injected post-activation failure','Unexpected activation error')
        else:raise ValueError('Post-activation fault did not fire')
    require(activated and snapshot()==baseline,'Activation failure rollback differs')
    return dict(passed=True,publication_rollback=rollback,write_failure_rolled_back=True,post_activation_failure_rolled_back=True,
        injected_after_actual_writes=writes,committed_catalog_mutations=0,
        publisher_sha256=sha(Path(publisher.__file__)),validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/two_body_revision_publication_transaction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,post_activation_failure_rolled_back=True,committed_catalog_mutations=0)))
