"""Exact stored legacy revision contracts and mandatory provenance checks."""
import importlib
import json
from scripts.stage_variance_identity_revisions import ROOT,history,revised_graph,sha,require
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg


def check(db,family,approved=False):
    imported=json.loads((ROOT/'docs/reviews/variance_identity_revision_import.json').read_text())
    require(imported['stager_sha256']==sha(ROOT/'scripts/stage_variance_identity_revisions.py'),'Legacy stager changed')
    require(imported['scope_audit_sha256']==sha(ROOT/'docs/reviews/variance_identity_execution_scope.json'),'Legacy scope evidence changed')
    items=[r for r in imported['graphs'] if r['family']==family]
    scopes=[r for r in json.loads((ROOT/'docs/reviews/variance_identity_execution_scope.json').read_text())['graphs'] if r['family']==family]
    require(len(items)==len(scopes)==1,'Unique family required')
    item=items[0];scope=scopes[0];version=item['version_id'];artifact=item['artifact_id']
    require(history(db,item['original_version_id'])==item['original_history_sha256'],'Legacy history changed')
    state=db.execute('SELECT fqdn,status,is_publishable FROM artifacts WHERE artifact_id=%s',(artifact,)).fetchone()
    require(state and state['status']==('approved' if approved else 'draft') and state['is_publishable']==approved,'Legacy approval differs')
    versions=db.execute('SELECT version_id::text,content_hash,is_latest,trust_tier,derives_from::text FROM artifact_versions WHERE artifact_id=%s',(artifact,)).fetchall()
    require([v['version_id'] for v in versions if v['is_latest']]==[version if approved else item['original_version_id']],'Legacy latest differs')
    require([v for v in versions if v['version_id']==version]==[dict(version_id=version,content_hash=item['graph_sha256'],is_latest=approved,trust_tier=3,derives_from=item['original_version_id'])],'Legacy version differs')
    from scripts.audit_variance_identity_scope import check_parent
    frozen=json.loads((ROOT/'docs/reviews/variance_identity_execution_scope.json').read_text())
    require(frozen['auditor_sha256']==sha(ROOT/'scripts/audit_variance_identity_scope.py') and frozen['parent_checker_sha256']==sha(ROOT/'scripts/check_variance_identity_parent.py'),'Variance scope/checker changed')
    parent,qualified=check_parent(db,scope['qualification'])
    require(qualified['version_id']==scope['approved_execution_version_id'] and qualified['graph_sha256']==scope['approved_execution_graph_sha256'],'Qualified family changed')
    expected=revised_graph(parent,scope)
    require(encode_execution_graph(expected)[0]==item['graph_sha256'],'Legacy projection differs')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(state['fqdn'],)).fetchone()['d']
    graph=_artifact_document_to_cdg(document,version_id=version,content_hash=item['graph_sha256'],require_execution_envelope=True)
    require(graph==expected,'Legacy stored graph differs')
    columns='node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary'
    bindings=db.execute(f'SELECT {columns} FROM artifact_cdg_bindings WHERE version_id=%s',(version,)).fetchall()
    parents=db.execute(f'SELECT {columns} FROM artifact_cdg_bindings WHERE version_id=%s',(qualified['version_id'],)).fetchall()
    require(len(parents)==1,'Qualified binding count differs')
    bound=dict(parents[0]);bound['evidence_summary']=dict(qualified_parent_evidence=bound['evidence_summary'],qualified_parent_version_id=qualified['version_id'],legacy_scope_sha256=imported['scope_audit_sha256'])
    require(bindings==[bound],'Legacy exact provider binding differs')
    ports=[]
    for direction,values in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
        for ordinal,p in enumerate(values):ports.append(dict(direction=direction,ordinal=ordinal,**{k:getattr(p,k) for k in ['name','type_desc','constraints','required','default_value_repr','dim_signature']}))
    require(db.execute('SELECT direction,ordinal,name,type_desc,constraints,required,default_value_repr,dim_signature FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()==ports,'Legacy port contract differs')
    deps=[]
    for selected in dict.fromkeys([item['original_version_id'],scope['projected_source_version_id'],qualified['version_id']]):
        target=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(selected,)).fetchone()
        deps.append(dict(dependency_artifact_fqdn=target['fqdn'],dependency_content_hash=target['content_hash'],dependency_role='cdg',optional=False,port_name='',
            binding_metadata=dict(source_version_id=selected,scope='Immutable legacy source, complete source-step projection and approved execution-family provenance; runtime uses direct provider binding.')))
    actual=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional,port_name,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s',(version,)).fetchall()
    require(sorted(actual,key=lambda r:r['dependency_content_hash'])==sorted(deps,key=lambda r:r['dependency_content_hash']),'Legacy provenance differs')
    require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(version,)).fetchone(),'Failed legacy audit')
    require(db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_id=%s',(artifact,)).fetchone()['n']==int(approved),'Legacy serving differs')
    return graph,dict(item,scope_audit_sha256=imported['scope_audit_sha256'])
