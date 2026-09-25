"""Rollback-only faults against each staged legacy physics revision."""
import json
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_sine_double_angle_identity_revisions import ROOT,sha,require,RUNNER
from scripts.check_sine_double_angle_identity_revision import check


def cases(item):
    version=item['version_id'];original=item['original_version_id'];artifact=item['artifact_id'];parent=item['qualified_family_version_id']
    return [
        ('parent_binding_hash','UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s',('0'*64,parent),'Sine double angle parent binding differs'),
        ('parent_approval',"UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s AND runner_version='sine_double_angle-corrected-community.v1'",(parent,),'Unique approved sine_double_angle targets required'),
        ('parent_provenance','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',(parent,),'Sine double angle parent provenance differs'),
        ('original_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,original),'Legacy history changed'),
        ('candidate_hash','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,version),'Legacy version differs'),
        ('candidate_tier','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(version,),'Legacy version differs'),
        ('candidate_lineage','UPDATE artifact_versions SET derives_from=NULL WHERE version_id=%s',(version,),'Legacy version differs'),
        ('missing_latest','UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(original,),'Legacy latest differs'),
        ('premature_approval',"UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s",(artifact,),'Legacy approval differs'),
        ('missing_binding','DELETE FROM artifact_cdg_bindings WHERE version_id=%s',(version,),'Legacy exact provider binding differs'),
        ('binding_hash','UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s',('0'*64,version),'Legacy exact provider binding differs'),
        ('binding_evidence',"UPDATE artifact_cdg_bindings SET evidence_summary='{}'::jsonb WHERE version_id=%s",(version,),'Legacy exact provider binding differs'),
        ('port_contract',"UPDATE artifact_io_specs SET constraints='corrupted' WHERE version_id=%s",(version,),'Legacy port contract differs'),
        ('optional_provenance','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',(version,),'Legacy provenance differs'),
        ('missing_provenance','DELETE FROM artifact_dependencies WHERE dependent_version_id=%s',(version,),'Legacy provenance differs'),
        ('failed_audit','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s',(version,),'Failed legacy audit'),
        ('execution_envelope',"UPDATE artifact_cdg_nodes SET type_signature='{}' WHERE version_id=%s",(version,),'execution version has no valid execution envelope'),
    ]


def validate():
    items=json.loads((ROOT/'docs/reviews/sine_double_angle_identity_revision_import.json').read_text())['graphs'];reports=[]
    connection=dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL']
    with psycopg.connect(connection,row_factory=dict_row,options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        for item in items:
            family=item['family'];baseline=check(db,family);rejected=[]
            for name,statement,params,expected in cases(item):
                db.execute('SAVEPOINT fault')
                try:
                    require(db.execute(statement,params).rowcount>0,'Fault changed no rows')
                    try:check(db,family)
                    except ValueError as error:require(str(error)==expected,'Unexpected '+family+'/'+name+' rejection: '+str(error));rejected.append(name)
                    else:raise ValueError('Fault accepted: '+family+'/'+name)
                finally:db.execute('ROLLBACK TO SAVEPOINT fault');db.execute('RELEASE SAVEPOINT fault')
                require(check(db,family)==baseline,'Fault rollback changed graph')
            reports.append(dict(family=family,version_id=item['version_id'],graph_sha256=item['graph_sha256'],rejected_faults=rejected))
        db.rollback()
    with psycopg.connect(connection,row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for item in items:check(db,item['family'])
    return dict(passed=True,approved=False,rollback_verified=True,committed_catalog_mutations=0,graphs=reports,
        validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_sine_double_angle_identity_revision.py'),
        scope='Fourteen candidate faults and three parent faults per original identity, plus intact reads after each rollback and in a fresh connection.')


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/sine_double_angle_identity_database_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,graphs=2,rejected_faults=sum(len(r['rejected_faults']) for r in report['graphs']),committed_catalog_mutations=0)))
