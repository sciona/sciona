"""Rollback-only corruption checks for the original projectile-resistance revision."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_projectile_revision import check
from scripts.stage_projectile_original_revision import ROOT,ARTIFACT,ORIGINAL,EXECUTION,sha,require


def cases(version):
    return [
        ('original_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,ORIGINAL),'Original history changed'),
        ('candidate_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,version),'Revision version differs'),
        ('candidate_tier','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(version,),'Revision version differs'),
        ('candidate_lineage','UPDATE artifact_versions SET derives_from=NULL WHERE version_id=%s',(version,),'Revision version differs'),
        ('missing_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(ORIGINAL,),'Latest version differs'),
        ('premature_approval',"UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s",(ARTIFACT,),'Revision approval differs'),
        ('parent_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,EXECUTION),'execution snapshot identity mismatch'),
        ('root_units',"UPDATE artifact_io_specs SET dim_signature='T2' WHERE version_id=%s AND direction='input'",(version,),'Stored port contract differs'),
        ('root_constraints',"UPDATE artifact_io_specs SET constraints='' WHERE version_id=%s",(version,),'Stored port contract differs'),
        ('missing_binding','DELETE FROM artifact_cdg_bindings WHERE version_id=%s',(version,),'Revision binding differs'),
        ('binding_hash','UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s',('0'*64,version),'Exact approved provider missing'),
        ('binding_evidence',"UPDATE artifact_cdg_bindings SET evidence_summary='{}'::jsonb WHERE version_id=%s",(version,),'Binding evidence differs'),
        ('provider_approval',"UPDATE artifacts SET status='draft',is_publishable=false WHERE fqdn=(SELECT bound_artifact_fqdn FROM artifact_cdg_bindings WHERE version_id=%s)",(version,),'Exact approved provider missing'),
        ('legacy_provider_hash',"UPDATE atom_versions SET content_hash=%s WHERE version_id=(SELECT (evidence_summary->>'provider_version_id')::uuid FROM artifact_cdg_bindings WHERE version_id=%s)",('0'*64,version),'Exact approved provider missing'),
        # Catalog triggers invalidate provider eligibility when legacy contracts change.
        ('legacy_provider_contract',"UPDATE atom_io_specs SET constraints='corrupted' WHERE version_id=(SELECT (evidence_summary->>'provider_version_id')::uuid FROM artifact_cdg_bindings WHERE version_id=%s)",(version,),'Exact approved provider missing'),
        ('optional_provenance','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',(version,),'Mandatory provenance differs'),
        ('missing_provenance','DELETE FROM artifact_dependencies WHERE dependent_version_id=%s',(version,),'Mandatory provenance differs'),
        ('failed_audit','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s',(version,),'Failed version audit'),
        ('envelope',"UPDATE artifact_cdg_nodes SET type_signature='{}' WHERE version_id=%s",(version,),'execution version has no valid execution envelope'),
    ]


def validate():
    imported=json.loads((ROOT/'docs/reviews/projectile_original_revision_import.json').read_text())
    version=imported['version_id'];rejected=[]
    connection=dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL']
    with psycopg.connect(connection,row_factory=dict_row,options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('projectile-original-revision-draft.v1'))")
        check(db)
        for name,statement,params,expected in cases(version):
            db.execute('SAVEPOINT fault')
            try:
                require(db.execute(statement,params).rowcount>0,'Fault changed no rows')
                try:check(db)
                except ValueError as error:
                    require(str(error)==expected,'Unexpected rejection for '+name+': '+str(error))
                    rejected.append(name)
                else:raise ValueError('Fault accepted: '+name)
            finally:
                db.execute('ROLLBACK TO SAVEPOINT fault');db.execute('RELEASE SAVEPOINT fault')
        check(db);db.rollback()
    with psycopg.connect(connection,row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:check(db)
    return dict(passed=True,approved=False,rollback_verified=True,committed_catalog_mutations=0,
        version_id=version,graph_sha256=imported['graph_sha256'],rejected_faults=rejected,
        validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_projectile_revision.py'),
        scope='Application catalog-integrity checker with rollback-only faults and a fresh read; not a database-trigger-only claim or publication authorization.')


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/projectile_revision_database_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rejected_faults=len(report['rejected_faults']),committed_catalog_mutations=0)))
