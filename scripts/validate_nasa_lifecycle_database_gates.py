"""Rollback-only integrity faults for paired lifecycle drafts and shared atoms."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_lifecycle_drafts import ROOT,plan,check_parent,BUILDERS,sha
from scripts.review_conditional_correction import SOURCE_HASH,SOURCE_VERSION,require


def check_staged(db,proposed,*,approved=False):
    check_parent(db,proposed['parent_domain'])
    targets=[(atom,True) for atom in proposed['atoms']]
    for graph in proposed['graphs'].values():
        targets.append((dict(graph,content_hash=graph['graph_sha256'],inputs=graph['boundary_inputs'],outputs=graph['boundary_outputs']),False))
    for target,is_atom in targets:
        tables = [('artifacts', 'artifact_versions', 'artifact_io_specs', 'artifact_id', 'draft')]
        if is_atom:
            tables.append(('atoms', 'atom_versions', 'atom_io_specs', 'atom_id', 'flagged'))
        for artifacts, versions, ports, key, status in tables:
            row = db.execute(f'SELECT fqdn,status,is_publishable FROM {artifacts} WHERE {key}=%s', (target['artifact_id'],)).fetchone()
            require(row == dict(fqdn=target['fqdn'], status='approved' if approved else status, is_publishable=approved), 'Draft identity/state differs')
            row = db.execute(f'SELECT {key} AS identity,content_hash,is_latest,trust_tier FROM {versions} WHERE version_id=%s',
                             (target['version_id'],)).fetchone()
            require(row and str(row['identity']) == target['artifact_id'] and row['content_hash'] == target['content_hash']
                    and row['is_latest'] == (is_atom or approved) and row['trust_tier'] == 3, 'Version identity/state differs')
            columns = ['direction', 'name', 'ordinal', 'type_desc', 'constraints', 'required', 'default_value_repr']
            if key == 'artifact_id':
                columns.append('dim_signature')
            actual = db.execute(f'SELECT {",".join(columns)} FROM {ports} WHERE version_id=%s ORDER BY direction,ordinal',
                                (target['version_id'],)).fetchall()
            expected = []
            for direction, selected in [('input', target['inputs']), ('output', target['outputs'])]:
                for ordinal, port in enumerate(selected):
                    expected.append({c: direction if c == 'direction' else ordinal if c == 'ordinal' else port[c] for c in columns})
            require(actual == expected, 'Stored port contracts differ')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',
                               (target['version_id'],)).fetchone(), 'Unresolved failed audit')
    parent=proposed['parent_domain']
    for kind,graph in proposed['graphs'].items():
        bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(graph['version_id'],)).fetchall()
        expected=[dict(node_id=b['node_id'],bound_artifact_fqdn=b['fqdn'],bound_version_content_hash=b['content_hash'],status='active',
            evidence_summary=dict(runtime_fqdn=b['runtime_fqdn'],provider_version_id=b['version_id'],output_aliases_by_ordinal=[p['name'] for p in b['outputs']]))
            for b in sorted(graph['bindings'],key=lambda b:b['node_id'])]
        require(bindings==expected,'Stored bindings differ')
        dependencies=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,optional,dependency_role,port_name,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn',(graph['version_id'],)).fetchall()
        expected=[]
        for fqdn,digest,version,scope in [
            (proposed['mandatory_provenance']['fqdn'],SOURCE_HASH,SOURCE_VERSION,'Mandatory original competition intake provenance; lifecycle reconstruction only.'),
            (parent['fqdn'],parent['graph_sha256'],parent['version_id'],'Mandatory approved domain structure provenance; shared atom bindings are invoked directly.')]:
            expected.append(dict(dependency_artifact_fqdn=fqdn,dependency_content_hash=digest,optional=False,dependency_role='cdg',port_name='',binding_metadata=dict(scope=scope,source_version_id=version)))
        require(dependencies==sorted(expected,key=lambda row:row['dependency_artifact_fqdn']),'Mandatory provenance differs')
        built=BUILDERS[kind]()
        count=db.execute('SELECT count(*) AS n FROM artifact_cdg_edges WHERE version_id=%s',(graph['version_id'],)).fetchone()['n']
        require(count==len(built.edges),'Stored edge count differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(graph['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=graph['version_id'],content_hash=graph['graph_sha256'],require_execution_envelope=True)==built,'Stored graph differs')


def fault_cases(proposed):
    faults=[]
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
            faults.extend([
                (kind+'_missing_binding_'+binding['node_id'],'DELETE FROM artifact_cdg_bindings WHERE version_id=%s AND node_id=%s',(version,binding['node_id']),'Stored bindings differ'),
                (kind+'_binding_hash_'+binding['node_id'],'UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s AND node_id=%s',('0'*64,version,binding['node_id']),'Stored bindings differ'),
            ])
    return faults


def validate():
    proposed=plan();rejected=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        for name in ['residual-classifier-draft.v1','nasa-domain-draft.v1','nasa-lifecycle-draft.v1']:
            db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(name,))
        check_staged(db,proposed)
        for name,statement,params,expected in fault_cases(proposed):
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
        db.rollback()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed)
    return dict(passed=True,approved=False,committed_catalog_mutations=0,rollback_verified=True,
        rejected_faults=rejected,graph_sha256={kind:graph['graph_sha256'] for kind,graph in proposed['graphs'].items()},
        validator_sha256=sha(Path(__file__)),scope='Application publication checker against rollback-only database faults; not a database-trigger-only claim.')


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/nasa_lifecycle_database_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rejected_faults=len(report['rejected_faults']),rollback_verified=True,committed_catalog_mutations=0)))
