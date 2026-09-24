"""Verify complete rollback and recovery after actual draft import writes."""
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import scripts.import_nasa_population_drafts as importer
from scripts.plan_nasa_population_drafts import ROOT,plan,check_parent
from scripts.review_conditional_correction import require,sha


def check_absent(proposed):
    atoms=proposed['atoms'];graphs=list(proposed['graphs'].values())
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_parent(db,proposed['parent_lifecycle'])
        for table,key,items,field in [('artifacts','artifact_id',atoms+graphs,'artifact_id'),
            ('atoms','atom_id',atoms,'artifact_id'),('artifact_versions','version_id',atoms+graphs,'version_id'),
            ('atom_versions','version_id',atoms,'version_id')]:
            require(not db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s::uuid[])',
                ([item[field] for item in items],)).fetchone(),'Draft rows survived rollback: '+table)
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE runner_version=%s',
            (importer.RUNNER,)).fetchone(),'Draft audit survived rollback')


def validate():
    proposed=plan()
    check_absent(proposed)
    rollback=importer.stage(False)
    require(rollback['rows_created']>0 and rollback['catalog_graph_roundtrips'],'Full staged graph check required')
    check_absent(proposed)
    original=importer.ensure_row;writes=0
    def fail_after_write(*args,**kwargs):
        nonlocal writes
        result=original(*args,**kwargs)
        writes+=int(result)
        if writes==3:raise RuntimeError('injected draft write failure')
        return result
    with patch.object(importer,'ensure_row',side_effect=fail_after_write):
        try:importer.stage(False)
        except RuntimeError as error:require(str(error)=='injected draft write failure','Unexpected staging error')
        else:raise ValueError('Draft fault did not fire')
    require(writes==3,'Fault must follow actual writes')
    check_absent(proposed)
    return dict(passed=True,full_rollback=rollback,injected_failure_rolled_back=True,
        injected_after_actual_writes=writes,committed_catalog_mutations=0,
        importer_sha256=sha(Path(importer.__file__)),validator_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/nasa_population_draft_transaction_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
