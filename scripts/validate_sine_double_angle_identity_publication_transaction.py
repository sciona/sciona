"""Rollback qualification for publication across both sine_double_angle identities."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_sine_double_angle_identity_revisions import ROOT,sha,require
from scripts.check_sine_double_angle_identity_revision import check
import scripts.promote_sine_double_angle_identity_revisions as publisher


def snapshot():
    items=json.loads((ROOT/'docs/reviews/sine_double_angle_identity_revision_import.json').read_text())['graphs'];ids=[r['artifact_id'] for r in items];result={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for item in items:check(db,item['family'])
        for table in ['artifacts','artifact_versions','artifact_descriptions','artifact_references','artifact_audit_rollups','artifact_audit_evidence','artifact_validity_bounds']:
            rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE artifact_id=ANY(%s::uuid[]) ORDER BY to_jsonb(t)::text',(ids,)).fetchall()
            result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    return result


def validate(source):
    before=snapshot();dry=publisher.promote(source)
    require(snapshot()==before,'Publication dry run changed catalog')
    original=publisher.ensure_row;writes=0
    def fail_after_write(*args,**kwargs):
        nonlocal writes
        result=original(*args,**kwargs);writes+=int(result)
        if writes==9:raise RuntimeError('injected cross-family publication failure')
        return result
    with patch.object(publisher,'ensure_row',side_effect=fail_after_write):
        try:publisher.promote(source)
        except RuntimeError as error:require(str(error)=='injected cross-family publication failure','Unexpected publication failure')
        else:raise ValueError('Cross-family failure did not fire')
    require(writes==9 and snapshot()==before,'Cross-family rollback differs')
    original_check=publisher.check_publication;activated=False
    def fail_after_activation(*args,**kwargs):
        nonlocal activated
        original_check(*args,**kwargs);activated=True
        raise RuntimeError('injected all-family activation failure')
    with patch.object(publisher,'check_publication',side_effect=fail_after_activation):
        try:publisher.promote(source)
        except RuntimeError as error:require(str(error)=='injected all-family activation failure','Unexpected activation failure')
        else:raise ValueError('Activation failure did not fire')
    require(activated and snapshot()==before,'All-family activation rollback differs')
    return dict(passed=True,publication_rollback=dry,cross_family_write_failure_rolled_back=True,post_activation_failure_rolled_back=True,
        injected_after_actual_writes=writes,committed_catalog_mutations=0,publisher_sha256=sha(publisher.__file__),validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/sine_double_angle_identity_publication_transaction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,post_activation_failure_rolled_back=True,committed_catalog_mutations=0)))
