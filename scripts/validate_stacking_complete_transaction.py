"""Verify complete-stack draft rollback, including failures after actual writes."""
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.stage_stacking_complete import ROOT, require, sha, stage


def snapshot():
    scope = json.loads((ROOT/'docs/reviews/competition_otto_decomposed_complete_scope.json').read_text())
    ids = [scope['artifact_id'], *[str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+fqdn))
                                 for fqdn in scope['providers']]]
    result = {}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for table in ('artifacts','artifact_versions','artifact_cdg_nodes','artifact_cdg_edges',
                      'artifact_cdg_bindings','artifact_io_specs','artifact_audit_evidence','artifact_validity_bounds',
                      'artifact_dependencies','artifact_audit_rollups','artifact_references',
                      'atoms','atom_versions','atom_io_specs','atom_audit_rollups','atom_references'):
            if table.startswith('atom_') or table=='atoms':
                predicate = 'atom_id=ANY(%s::uuid[])'
            elif table in ('artifact_cdg_nodes','artifact_cdg_edges','artifact_cdg_bindings','artifact_dependencies'):
                key = 'dependent_version_id' if table=='artifact_dependencies' else 'version_id'
                predicate = key+' IN (SELECT version_id FROM artifact_versions WHERE artifact_id=ANY(%s::uuid[]))'
            else:
                predicate = 'artifact_id=ANY(%s::uuid[])'
            rows = db.execute('SELECT to_jsonb(t) AS row FROM '+table+' t WHERE '+predicate+
                              ' ORDER BY to_jsonb(t)::text', (ids,)).fetchall()
            result[table] = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    return result


def main():
    before = snapshot()
    dry = stage()
    require(dry['rows_created']>0 and dry['providers']==27 and dry['catalog_roundtrip'], 'Absent draft baseline required')
    require(snapshot()==before, 'Dry-run rollback changed catalog')
    points = [1, dry['rows_created']//2, dry['rows_created']]
    for point in points:
        try:
            stage(fail_after_writes=point)
        except RuntimeError as error:
            require(str(error)=='injected complete-stack staging failure', 'Unexpected staging failure')
        else:
            raise ValueError('Injected failure did not fire')
        require(snapshot()==before, 'Injected failure changed catalog')
    report = dict(passed=True, rollback_verified=True, injected_failures_rolled_back=points,
                  committed_catalog_mutations=0, snapshot_tables=len(before), draft=dry,
                  stager_sha256=sha(ROOT/'scripts/stage_stacking_complete.py'), validator_sha256=sha(__file__),
                  scope_sha256=sha(ROOT/'docs/reviews/competition_otto_decomposed_complete_scope.json'),
                  limitations=['Draft transaction qualification only; no approval or served execution.'])
    (ROOT/'docs/reviews/competition_otto_decomposed_draft_transaction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True, rollback_points=points, committed_catalog_mutations=0)))


if __name__=='__main__':
    main()
