"""Check the corrected version and preserved history independently of old draft-only audits."""
from sciona.nasa_complete_workflow_graph import build_complete_workflow_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_corrected_intake import check_source_anchor,check_atoms,check_parents,origin_rows,require

PUBLICATION_RUNNER='nasa-corrected-intake-community.v1'


def check_candidate(db,proposed,*,approved=False):
    check_source_anchor(db)
    require(origin_rows(db)==proposed['original_rows_sha256'],'Original history differs')
    check_atoms(db,proposed['atoms']);check_parents(db,proposed['parent_graphs'])
    row=db.execute('SELECT fqdn,status,is_publishable FROM artifacts WHERE artifact_id=%s',(proposed['artifact_id'],)).fetchone()
    require(row==dict(fqdn=proposed['fqdn'],status='approved' if approved else 'draft',is_publishable=approved),'Intake approval state differs')
    versions=db.execute('SELECT version_id,content_hash,is_latest,trust_tier FROM artifact_versions WHERE artifact_id=%s',(proposed['artifact_id'],)).fetchall()
    selected=[row for row in versions if str(row['version_id'])==proposed['version_id']]
    require(len(selected)==1 and selected[0]['content_hash']==proposed['graph_sha256'] and selected[0]['trust_tier']==3
        and selected[0]['is_latest']==approved,'Corrected version differs')
    require([str(row['version_id']) for row in versions if row['is_latest']]==[
        proposed['version_id'] if approved else proposed['original_version_id']],'Intake latest-version selection differs')
    columns=['direction','name','ordinal','type_desc','constraints','required','default_value_repr','dim_signature']
    actual=db.execute(f'SELECT {",".join(columns)} FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(proposed['version_id'],)).fetchall()
    expected=[]
    for direction,ports in [('input',proposed['inputs']),('output',proposed['outputs'])]:
        for ordinal,port in enumerate(ports):
            expected.append({name:direction if name=='direction' else ordinal if name=='ordinal' else port[name] for name in columns})
    require(actual==expected,'Corrected root contracts differ')
    actual=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(proposed['version_id'],)).fetchall()
    expected=[dict(node_id=b['node_id'],bound_artifact_fqdn=b['fqdn'],bound_version_content_hash=b['content_hash'],status='active',
        evidence_summary=dict(runtime_fqdn=b['runtime_fqdn'],provider_version_id=b['version_id'],output_aliases_by_ordinal=[p['name'] for p in b['outputs']]))
        for b in sorted(proposed['bindings'],key=lambda b:b['node_id'])]
    require(actual==expected,'Corrected bindings differ')
    sources=[dict(fqdn=proposed['fqdn'],graph_sha256=proposed['original_content_hash'],version_id=proposed['original_version_id'])]
    sources+=list(proposed['parent_graphs'].values())
    expected=sorted([dict(dependency_artifact_fqdn=s['fqdn'],dependency_content_hash=s['graph_sha256'],dependency_role='cdg',optional=False,port_name='',
        binding_metadata=dict(source_version_id=s['version_id'],scope='Immutable source/version provenance; runtime computation is expressed by direct atom bindings.'))
        for s in sources],key=lambda row:row['dependency_artifact_fqdn'])
    actual=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional,port_name,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn',(proposed['version_id'],)).fetchall()
    require(actual==expected,'Corrected provenance differs')
    require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(proposed['version_id'],)).fetchone(),'Unresolved corrected-version failure')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
    restored=_artifact_document_to_cdg(document,version_id=proposed['version_id'],content_hash=proposed['graph_sha256'],require_execution_envelope=True)
    require(restored==build_complete_workflow_graph(10),'Corrected graph serialization differs')
    served=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_id=%s',(proposed['artifact_id'],)).fetchone()['n']
    require(served==int(approved),'Corrected serving state differs')
    if approved:
        require(db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s AND audit_type='semantic_audit' AND passed",
            (proposed['version_id'],PUBLICATION_RUNNER)).fetchone(),'Corrected-version publication evidence required')
