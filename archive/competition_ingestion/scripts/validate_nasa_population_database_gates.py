"""Rollback-only integrity faults for paired lifecycle drafts and shared atoms."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_population_drafts import ROOT,plan,check_parent,BUILDERS,sha
from scripts.review_conditional_correction import require
from scripts.check_nasa_population_catalog import check_staged


def fault_cases(proposed):
    faults=[]
    covered_providers=set()
    for index,atom in enumerate(proposed['atoms']):
        version=atom['version_id']
        faults.extend([
            (f'provider_hash_{index}','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,version),'Version identity/state differs'),
            (f'provider_tier_{index}','UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s',(version,),'Version identity/state differs'),
            (f'legacy_hash_{index}','UPDATE atom_versions SET content_hash=%s WHERE version_id=%s',('0'*64,version),'Version identity/state differs'),
            (f'legacy_contract_{index}',"UPDATE atom_io_specs SET constraints='' WHERE version_id=%s AND name=%s",(version,atom['inputs'][0]['name']),'Stored port contracts differ'),
        ])
    for index,atom in enumerate(proposed['reused_atoms']):
        faults.extend([
            (f'reused_hash_{index}','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,atom['version_id']),'Version identity/state differs'),
            (f'reused_approval_{index}',"UPDATE artifacts SET status='draft',is_publishable=false WHERE artifact_id=%s",(atom['artifact_id'],),'Draft identity/state differs'),
        ])
    for kind,graph in proposed['graphs'].items():
        version=graph['version_id'];edge=BUILDERS[kind]().edges[0]
        faults.extend([
            (kind+'_latest','UPDATE artifact_versions SET is_latest=true WHERE version_id=%s',(version,),'Version identity/state differs'),
            (kind+'_contract',"UPDATE artifact_io_specs SET constraints='' WHERE version_id=%s AND name=%s",(version,graph['boundary_inputs'][0]['name']),'Stored port contracts differ'),
            (kind+'_optional_provenance','UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s',(version,),'Mandatory provenance differs'),
            (kind+'_provenance_metadata',"UPDATE artifact_dependencies SET binding_metadata='{}'::jsonb WHERE dependent_version_id=%s",(version,),'Mandatory provenance differs'),
            (kind+'_failed_audit','UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s',(version,),'Unresolved failed audit'),
            (kind+'_runtime_binding',"UPDATE artifact_cdg_bindings SET evidence_summary='{}'::jsonb WHERE version_id=%s",(version,),'Stored bindings differ'),
            (kind+'_missing_edge','DELETE FROM artifact_cdg_edges WHERE version_id=%s AND source_id=%s AND target_id=%s AND input_name=%s',(version,edge.source_id,edge.target_id,edge.input_name),'Stored edge count differs'),
        ])
        for binding in graph['bindings']:
            if binding['runtime_fqdn'] in covered_providers:continue
            covered_providers.add(binding['runtime_fqdn'])
            faults.extend([
                (kind+'_missing_binding_'+binding['node_id'],'DELETE FROM artifact_cdg_bindings WHERE version_id=%s AND node_id=%s',(version,binding['node_id']),'Stored bindings differ'),
                (kind+'_binding_hash_'+binding['node_id'],'UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s AND node_id=%s',('0'*64,version,binding['node_id']),'Stored bindings differ'),
            ])
    return faults


def validate():
    proposed=plan();rejected=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        for name in ['residual-classifier-draft.v1','nasa-domain-draft.v1','nasa-lifecycle-draft.v1','nasa-population-draft.v1']:
            db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(name,))
        check_staged(db,proposed)
        cases=fault_cases(proposed)
        for index,(name,statement,params,expected) in enumerate(cases):
            db.execute('SAVEPOINT fault')
            try:
                require(db.execute(statement,params).rowcount>0,'Fault injection changed no rows')
                try:check_staged(db,proposed)
                except ValueError as error:
                    require(str(error)==expected,'Unexpected rejection: '+str(error))
                    rejected.append(name)
                else:raise ValueError('Fault accepted: '+name)
            finally:
                db.execute('ROLLBACK TO SAVEPOINT fault')
                db.execute('RELEASE SAVEPOINT fault')
            check_staged(db,proposed)
            if (index+1)%25==0:print(json.dumps(dict(rejected=index+1,total=len(cases))),flush=True)
        db.rollback()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed)
    return dict(passed=True,approved=False,committed_catalog_mutations=0,rollback_verified=True,
        rejected_faults=rejected,graph_sha256={kind:graph['graph_sha256'] for kind,graph in proposed['graphs'].items()},
        validator_sha256=sha(Path(__file__)),checker_sha256=sha(ROOT/'scripts/check_nasa_population_catalog.py'),
        binding_fault_scope='One missing/hash pair per distinct provider across the family, plus runtime metadata corruption in every graph; the checker compares every stored binding.',scope='Application publication checker against rollback-only database faults; not a database-trigger-only claim.')


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/nasa_population_database_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rejected_faults=len(report['rejected_faults']),rollback_verified=True,committed_catalog_mutations=0)))
