"""Verify all-family staging rollback and failure after one complete family."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_sine_double_angle_identity_revisions import ROOT,stage,sha,require


def snapshot():
    scope=json.loads((ROOT/'docs/reviews/sine_double_angle_identity_execution_scope.json').read_text())
    ids=[r['legacy_artifact_id'] for r in scope['graphs']]
    result={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for table in ['artifacts','artifact_versions','artifact_cdg_nodes','artifact_cdg_edges','artifact_cdg_bindings','artifact_io_specs','artifact_audit_evidence','artifact_validity_bounds','artifact_dependencies']:
            if table in ['artifacts','artifact_versions','artifact_io_specs','artifact_audit_evidence','artifact_validity_bounds']:
                where='artifact_id=ANY(%s::uuid[])'
            else:
                key='dependent_version_id' if table=='artifact_dependencies' else 'version_id'
                where=f'{key} IN (SELECT version_id FROM artifact_versions WHERE artifact_id=ANY(%s::uuid[]))'
            rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {where} ORDER BY to_jsonb(t)::text',(ids,)).fetchall()
            result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    return result


def validate(source):
    before=snapshot();dry=stage(source)
    require(len(dry['graphs'])==2 and all(r['rows_created']>0 for r in dry['graphs']),'Absent-candidate baseline required')
    require(snapshot()==before,'Full staging rollback changed catalog')
    failure=dry['graphs'][0]['rows_created']+3
    try:stage(source,fail_after_writes=failure)
    except RuntimeError as error:require(str(error)=='injected legacy staging failure','Unexpected transaction error')
    else:raise ValueError('Injected write failure did not fire')
    require(snapshot()==before,'Cross-family rollback changed catalog')
    return dict(passed=True,rollback_verified=True,injected_failure_rollback_verified=True,
        injected_after_actual_writes=failure,completed_families_before_injected_failure=1,committed_catalog_mutations=0,
        draft=dry,stager_sha256=sha(ROOT/'scripts/stage_sine_double_angle_identity_revisions.py'),validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/sine_double_angle_identity_revision_transaction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,graphs=2,committed_catalog_mutations=0,injected_after_actual_writes=report['injected_after_actual_writes'])))
