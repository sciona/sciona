"""Exercise draft publication invariants using rollback-only fault injection."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.nasa_domain_graph import build_nasa_domain_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_domain_draft import plan, check_reused
from scripts.review_conditional_correction import ROOT, SOURCE_HASH, SOURCE_VERSION, require, sha


def check_staged(db, proposed, *, approved=False):
    check_reused(db, proposed['numerical_core'])
    source = db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v '
                       'USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
    require(source and source['content_hash'] == SOURCE_HASH and source['status'] == 'draft' and not source['is_publishable'],
            'Source provenance differs')
    graph = dict(artifact_id=proposed['artifact_id'], version_id=proposed['version_id'], fqdn=proposed['fqdn'],
                 content_hash=proposed['graph_sha256'], inputs=proposed['boundary_inputs'], outputs=proposed['boundary_outputs'])
    for target in proposed['atoms'] + [graph]:
        is_atom = target is not graph
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
    bindings = db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings '
                          'WHERE version_id=%s ORDER BY node_id', (proposed['version_id'],)).fetchall()
    expected = [dict(node_id=b['node_id'], bound_artifact_fqdn=b['fqdn'], bound_version_content_hash=b['content_hash'],
                     status='active', evidence_summary=dict(runtime_fqdn=b['runtime_fqdn'], provider_version_id=b['version_id'],
                         output_aliases_by_ordinal=[p['name'] for p in b['outputs']]))
                for b in sorted(proposed['bindings'], key=lambda b: b['node_id'])]
    require(bindings == expected, 'Stored bindings differ')
    dependencies = db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,optional,dependency_role,port_name '
                              'FROM artifact_dependencies WHERE dependent_version_id=%s', (proposed['version_id'],)).fetchall()
    core=proposed['numerical_core']
    expected_dependencies=[dict(dependency_artifact_fqdn=source['fqdn'], dependency_content_hash=SOURCE_HASH,
                               optional=False,dependency_role='cdg',port_name=''),
                           dict(dependency_artifact_fqdn=core['fqdn'],dependency_content_hash=core['graph_sha256'],
                               optional=False,dependency_role='cdg',port_name='')]
    require(sorted(dependencies,key=lambda row:row['dependency_artifact_fqdn']) ==
            sorted(expected_dependencies,key=lambda row:row['dependency_artifact_fqdn']), 'Mandatory provenance differs')
    metadata=db.execute('SELECT binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s '
        'AND dependency_artifact_fqdn=%s',(proposed['version_id'],core['fqdn'])).fetchone()['binding_metadata']
    require(metadata==dict(scope='Mandatory approved numerical structure provenance; its shared atom bindings are invoked directly.',
                           source_version_id=core['version_id']), 'Numerical provenance metadata differs')
    count=db.execute('SELECT count(*) AS n FROM artifact_cdg_edges WHERE version_id=%s',(proposed['version_id'],)).fetchone()['n']
    require(count==len(build_nasa_domain_graph().edges),'Stored edge count differs')
    document = db.execute('SELECT get_artifact_document(%s) AS d', (proposed['fqdn'],)).fetchone()['d']
    require(_artifact_document_to_cdg(document, version_id=proposed['version_id'], content_hash=proposed['graph_sha256'],
            require_execution_envelope=True) == build_nasa_domain_graph(), 'Stored graph differs')


def validate():
    proposed = plan()
    graph, atom = proposed['version_id'], proposed['bindings'][0]['version_id']
    faults = [
        ('graph_latest', 'UPDATE artifact_versions SET is_latest=true WHERE version_id=%s', (graph,), 'Version identity/state differs'),
        ('graph_contract', "UPDATE artifact_io_specs SET constraints='' WHERE version_id=%s AND name='maximum_error'", (graph,), 'Stored port contracts differ'),
        ('runtime_binding', "UPDATE artifact_cdg_bindings SET evidence_summary='{}'::jsonb WHERE version_id=%s", (graph,), 'Stored bindings differ'),
        ('optional_provenance', 'UPDATE artifact_dependencies SET optional=true WHERE dependent_version_id=%s', (graph,), 'Mandatory provenance differs'),
        ('failed_audit', 'UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s', (graph,), 'Unresolved failed audit'),
    ]
    for index, target in enumerate(proposed['atoms']):
        version = target['version_id']
        first_input = target['inputs'][0]['name']
        faults.extend([
            (f'provider_hash_{index}', 'UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s', ('0'*64, version), 'Version identity/state differs'),
            (f'provider_tier_{index}', 'UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s', (version,), 'Version identity/state differs'),
            (f'legacy_hash_{index}', 'UPDATE atom_versions SET content_hash=%s WHERE version_id=%s', ('0'*64, version), 'Version identity/state differs'),
            (f'legacy_contract_{index}', "UPDATE atom_io_specs SET constraints='' WHERE version_id=%s AND name=%s", (version, first_input), 'Stored port contracts differ'),
        ])
    for binding in proposed['bindings']:
        faults.extend([
            ('missing_binding_'+binding['node_id'], 'DELETE FROM artifact_cdg_bindings WHERE version_id=%s AND node_id=%s', (graph, binding['node_id']), 'Stored bindings differ'),
            ('binding_hash_'+binding['node_id'], 'UPDATE artifact_cdg_bindings SET bound_version_content_hash=%s WHERE version_id=%s AND node_id=%s', ('0'*64, graph, binding['node_id']), 'Stored bindings differ'),
        ])
    faults.append(('removed_handoff_edge',"DELETE FROM artifact_cdg_edges WHERE version_id=%s AND source_id='domain_handoff' AND target_id='training' AND input_name='features'",(graph,),'Stored edge count differs'))
    faults.append(('numerical_provenance_metadata',"UPDATE artifact_dependencies SET binding_metadata='{}'::jsonb WHERE dependent_version_id=%s AND dependency_artifact_fqdn=%s",(graph,proposed['numerical_core']['fqdn']),'Numerical provenance metadata differs'))
    for index,target in enumerate(proposed['reused_atoms']):
        faults.extend([
            (f'reused_hash_{index}','UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s',('0'*64,target['version_id']),'Version identity/state differs'),
            (f'reused_approval_{index}',"UPDATE artifacts SET status='draft',is_publishable=false WHERE artifact_id=%s",(target['artifact_id'],),'Draft identity/state differs'),
        ])
    rejected = []
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('residual-classifier-draft.v1'))")
        db.execute("SELECT pg_advisory_xact_lock(hashtext('nasa-domain-draft.v1'))")
        check_staged(db, proposed)
        for name, statement, params, expected in faults:
            db.execute('SAVEPOINT injected_fault')
            try:
                changed = db.execute(statement, params).rowcount
                require(changed > 0, 'Fault injection did not change any row')
                try:
                    check_staged(db, proposed)
                except ValueError as error:
                    require(str(error) == expected, 'Unexpected rejection: ' + str(error))
                    rejected.append(name)
                else:
                    raise ValueError('Fault accepted: ' + name)
            finally:
                db.execute('ROLLBACK TO SAVEPOINT injected_fault')
                db.execute('RELEASE SAVEPOINT injected_fault')
            check_staged(db, proposed)
        db.rollback()
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db, proposed)
    return dict(format='nasa-domain-database-gates.v1', passed=True, rejected_faults=rejected,
        rollback_verified=True, committed_catalog_mutations=0, graph_sha256=proposed['graph_sha256'],
        validator_sha256=sha(Path(__file__)), approved=False,
        scope='Application prepublication integrity checker exercised against injected database faults; not a claim that database triggers alone enforce these checks. License/runtime publication review and approval transaction remain pending.')


if __name__ == '__main__':
    report = validate()
    (ROOT / 'docs/reviews/nasa_domain_database_gates.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
