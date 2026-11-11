"""Rollback-only faults against every decomposed Otto provider and graph contract."""
import json
from uuid import NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_stacking_complete_catalog import ROOT, qualification, check_candidate, require, sha
from scripts.stage_stacking_complete import RUNNER
from scripts.validate_stacking_complete_transaction import snapshot


def fault_cases(scope):
    cases = []
    version, original = scope['corrected_version_id'], scope['original_version_id']

    def add(name, statement, parameters, message):
        cases.append((name, [(statement,parameters)], message))

    add('original_identity', "UPDATE artifacts SET fqdn=fqdn||'.fault' WHERE artifact_id=%s",
        (scope['artifact_id'],), 'Original identity state differs')
    add('original_history', "UPDATE artifact_io_specs SET constraints=constraints||' fault' WHERE version_id=%s",
        (original,), 'Original history differs')
    add('original_snapshot', "UPDATE artifact_audit_evidence SET details='{}'::jsonb WHERE version_id=%s AND runner_version='competition-intake.v1'",
        (original,), 'Original history differs')
    add('original_latest', 'UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',
        (original,), 'Original latest selection differs')
    add('premature_approval', "UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s",
        (scope['artifact_id'],), 'Original identity state differs')
    for name, column, value in [('candidate_hash','content_hash','0'*64),('candidate_tier','trust_tier',1),
                                ('candidate_lineage','derives_from',None)]:
        add(name,'UPDATE artifact_versions SET '+column+'=%s WHERE version_id=%s',(value,version),
            'Corrected version differs')
    cases.append(('premature_latest', [
        ('UPDATE artifact_versions SET is_latest=false WHERE version_id=%s',(original,)),
        ('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s',(version,))], 'Corrected version differs'))
    add('root_contract', "UPDATE artifact_io_specs SET required=false WHERE version_id=%s AND direction='input'",
        (version,), 'Corrected root contracts differ')
    add('missing_binding', "DELETE FROM artifact_cdg_bindings WHERE version_id=%s AND node_id='unpack'",
        (version,), 'Corrected bindings differ')
    add('binding_hash', "UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s AND node_id='unpack'",
        ('0'*64,version), 'Corrected bindings differ')
    add('binding_metadata', "UPDATE artifact_cdg_bindings SET evidence_summary='{}'::jsonb WHERE version_id=%s AND node_id='unpack'",
        (version,), 'Corrected bindings differ')
    add('optional_provenance','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',
        (version,), 'Corrected provenance differs')
    add('provenance_metadata',"UPDATE artifact_dependencies SET binding_metadata='{}'::jsonb WHERE dependent_version_id=%s",
        (version,), 'Corrected provenance differs')
    add('staging_evidence',"UPDATE artifact_audit_evidence SET details='{}'::jsonb WHERE version_id=%s AND runner_version=%s",
        (version,RUNNER), 'Corrected staging evidence differs')
    add('failed_staging', 'UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s AND runner_version=%s',
        (version,RUNNER), 'Corrected staging evidence differs')
    add('graph_edge', "DELETE FROM artifact_cdg_edges WHERE version_id=%s AND target_id='embedding' AND input_name='seed'",
        (version,), 'execution graph projection drift')
    for index,(fqdn,p) in enumerate(sorted(scope['providers'].items())):
        owner = str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+fqdn))
        selected = p['version_id']
        for table,key in [('artifacts','artifact_id'),('atoms','atom_id')]:
            add(f'provider_{index}_{table}_approval', 'UPDATE '+table+" SET status='approved',is_publishable=true WHERE "+key+'=%s',
                (owner,), 'Provider identity differs: '+table)
        add(f'provider_{index}_implementation',"UPDATE atoms SET source_symbol=source_symbol||'_fault' WHERE atom_id=%s",
            (owner,), 'Provider identity differs: atoms')
        for table in ('artifact_versions','atom_versions'):
            add(f'provider_{index}_{table}_hash','UPDATE '+table+' SET content_hash=%s WHERE version_id=%s',
                ('0'*64,selected), 'Provider version differs: '+table)
            add(f'provider_{index}_{table}_tier','UPDATE '+table+' SET trust_tier=1 WHERE version_id=%s',
                (selected,), 'Provider version differs: '+table)
        for table in ('artifact_io_specs','atom_io_specs'):
            add(f'provider_{index}_{table}_ports', 'UPDATE '+table+" SET type_desc=type_desc||' fault' WHERE version_id=%s",
                (selected,), 'Provider ports differ: '+table)
        add(f'provider_{index}_evidence',"UPDATE artifact_audit_evidence SET details='{}'::jsonb WHERE version_id=%s AND runner_version=%s",
            (selected,RUNNER), 'Provider staging evidence differs')
    return cases


def main():
    scope,specs = qualification(); before=snapshot(); rejected=[]
    url=dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL']
    with psycopg.connect(url,row_factory=dict_row,options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        check_candidate(db,scope,specs)
        cases=fault_cases(scope)
        for name,actions,expected in cases:
            db.execute('SAVEPOINT fault')
            try:
                for statement,parameters in actions:
                    require(db.execute(statement,parameters).rowcount>0,'Fault changed no rows: '+name)
                try:
                    check_candidate(db,scope,specs)
                except ValueError as error:
                    require(expected in str(error),'Unexpected rejection for '+name+': '+str(error))
                    rejected.append(name)
                else:
                    raise ValueError('Catalog fault accepted: '+name)
            finally:
                db.execute('ROLLBACK TO SAVEPOINT fault')
                db.execute('RELEASE SAVEPOINT fault')
            if len(rejected)%25==0:
                print(json.dumps(dict(rejected=len(rejected),total=len(cases))),flush=True)
        check_candidate(db,scope,specs)
        db.rollback()
    require(snapshot()==before,'Fault validation changed catalog')
    with psycopg.connect(url,row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,scope,specs)
    require(qualification()[0]==scope,'Qualification changed during fault validation')
    report=dict(passed=True,approved=False,committed_catalog_mutations=0,rollback_verified=True,
        nodes=62,providers=27,rejected_faults=rejected,version_id=scope['corrected_version_id'],
        graph_sha256=scope['graph_sha256'],checker_sha256=sha(ROOT/'scripts/check_stacking_complete_catalog.py'),
        validator_sha256=sha(__file__),scope_sha256=sha(ROOT/'docs/reviews/competition_otto_decomposed_complete_scope.json'),
        limitations=['Draft integrity qualification only; no publication or served numerical execution.',
                     'All injected changes rolled back; existing failed evidence remains unchanged.'])
    (ROOT/'docs/reviews/competition_otto_decomposed_catalog_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rejected_faults=len(rejected),committed_catalog_mutations=0)))


if __name__=='__main__':
    main()
