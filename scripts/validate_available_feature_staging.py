"""Exercise real draft writes and rollback at early, middle and final writes."""
import hashlib
import json

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.plan_available_feature_providers import ROOT,plan,sha
from scripts.stage_available_feature_providers import stage


def snapshot(proposed):
    identities=[a['artifact_id'] for a in proposed['atoms']]+[proposed['source_provenance']['artifact_id']]
    versions=[a['version_id'] for a in proposed['atoms']]+[proposed['source_provenance']['version_id']]
    result={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for table,key,values in [('artifacts','artifact_id',identities),('atoms','atom_id',identities),
                ('artifact_versions','version_id',versions),('atom_versions','version_id',versions),
                ('artifact_io_specs','version_id',versions),('atom_io_specs','version_id',versions),
                ('artifact_audit_evidence','version_id',versions),('artifact_cdg_nodes','version_id',versions),
                ('artifact_cdg_edges','version_id',versions),('artifact_cdg_bindings','version_id',versions),
                ('artifact_dependencies','dependent_version_id',versions)]:
            rows=db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s::uuid[]) ORDER BY to_jsonb(t)::text',(values,)).fetchall()
            result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def validate():
    proposed=plan();before=snapshot(proposed)
    dry=stage()
    if dry['rows_created']!=265 or snapshot(proposed)!=before:raise ValueError('Dry-run writes or rollback differ')
    failures=[]
    for write in [1,132,265]:
        try:stage(apply=True,fault_after=write)
        except RuntimeError as error:
            if str(error)!='Injected draft write failure':raise
        else:raise ValueError('Expected injected failure did not occur')
        if snapshot(proposed)!=before:raise ValueError('Failed transaction changed catalog state')
        failures.append(write)
    return dict(passed=True,approved=False,catalog_mutations=0,draft_atoms=13,dry_run_rows=265,
        failure_after_write_checks=failures,original_and_target_rows_unchanged=True,
        implementation_sha256={path:sha(ROOT/path) for path in ['scripts/plan_available_feature_providers.py',
            'scripts/stage_available_feature_providers.py','scripts/validate_available_feature_staging.py','sciona/available_feature_contracts.py']})


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/available_feature_provider_staging_gates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['passed','dry_run_rows','failure_after_write_checks','original_and_target_rows_unchanged']}))
