"""Exact catalog integrity checks for population graphs and approved shared atoms."""
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_population_drafts import check_parent,BUILDERS
from scripts.review_conditional_correction import SOURCE_HASH,SOURCE_VERSION,require


def check_staged(db,proposed,*,approved=False):
    check_parent(db,proposed['parent_lifecycle'])
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
    for kind,graph in proposed['graphs'].items():
        parent=proposed['parent_lifecycle']['graphs'][kind.split('_')[0]]
        bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(graph['version_id'],)).fetchall()
        expected=[dict(node_id=b['node_id'],bound_artifact_fqdn=b['fqdn'],bound_version_content_hash=b['content_hash'],status='active',
            evidence_summary=dict(runtime_fqdn=b['runtime_fqdn'],provider_version_id=b['version_id'],output_aliases_by_ordinal=[p['name'] for p in b['outputs']]))
            for b in sorted(graph['bindings'],key=lambda b:b['node_id'])]
        require(bindings==expected,'Stored bindings differ')
        dependencies=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,optional,dependency_role,port_name,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn',(graph['version_id'],)).fetchall()
        expected=[]
        for fqdn,digest,version,scope in [
            (proposed['mandatory_provenance']['fqdn'],SOURCE_HASH,SOURCE_VERSION,'Mandatory original competition intake provenance; explicit population reconstruction only.'),
            (parent['fqdn'],parent['graph_sha256'],parent['version_id'],'Mandatory approved lifecycle structure provenance; each population branch invokes the shared atoms directly.')]:
            expected.append(dict(dependency_artifact_fqdn=fqdn,dependency_content_hash=digest,optional=False,dependency_role='cdg',port_name='',binding_metadata=dict(scope=scope,source_version_id=version)))
        require(dependencies==sorted(expected,key=lambda row:row['dependency_artifact_fqdn']),'Mandatory provenance differs')
        built=BUILDERS[kind]()
        count=db.execute('SELECT count(*) AS n FROM artifact_cdg_edges WHERE version_id=%s',(graph['version_id'],)).fetchone()['n']
        require(count==len(built.edges),'Stored edge count differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(graph['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=graph['version_id'],content_hash=graph['graph_sha256'],require_execution_envelope=True)==built,'Stored graph differs')
