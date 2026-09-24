"""Version, provenance, implementation and port checks for the original revision."""
import inspect
import json

from scripts.stage_period_frequency_original_revision import ROOT,ARTIFACT,ORIGINAL,REPLAY,EXECUTION,history,sha,require
from sciona.atoms.physical_quantities.period_frequency import frequency_from_period
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph


def check(db,approved=False):
    imported=json.loads((ROOT/'docs/reviews/period_frequency_original_revision_import.json').read_text())
    require(imported['stager_sha256']==sha(ROOT/'scripts/stage_period_frequency_original_revision.py'),'Stager changed')
    require(imported['scope_audit_sha256']==sha(ROOT/'docs/reviews/physics_period_frequency_original_scope.json'),'Scope evidence changed')
    require(history(db)==imported['original_history_sha256'],'Original history changed')
    version=imported['version_id'];digest=imported['graph_sha256']
    state=db.execute('SELECT fqdn,status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
    require(state and state['status']==('approved' if approved else 'draft') and state['is_publishable']==approved,'Revision approval differs')
    versions=db.execute('SELECT version_id::text,content_hash,is_latest,trust_tier,derives_from::text FROM artifact_versions WHERE artifact_id=%s',(ARTIFACT,)).fetchall()
    require([v['version_id'] for v in versions if v['is_latest']]==[version if approved else ORIGINAL],'Latest version differs')
    selected=[v for v in versions if v['version_id']==version]
    require(selected==[dict(version_id=version,content_hash=digest,is_latest=approved,trust_tier=3,derives_from=REPLAY)],'Revision version differs')
    parent=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(EXECUTION,)).fetchone()
    require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Parent approval differs')
    parent_document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
    expected=_artifact_document_to_cdg(parent_document,version_id=EXECUTION,content_hash=parent['content_hash'],require_execution_envelope=True).model_copy(deep=True)
    expected.metadata.update(original_artifact_id=ARTIFACT,original_version_id=ORIGINAL,qualified_numerical_parent_version_id=EXECUTION,
        revision_scope='Executable terminal relation under original identity; original algebraic derivation remains immutable provenance.')
    require(encode_execution_graph(expected)[0]==digest,'Candidate projection differs')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(state['fqdn'],)).fetchone()['d']
    graph=_artifact_document_to_cdg(document,version_id=version,content_hash=digest,require_execution_envelope=True)
    require(graph==expected,'Stored execution differs')
    bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s',(version,)).fetchall()
    require(len(bindings)==1 and bindings[0]['node_id']=='frequency' and bindings[0]['status']=='active'
        and bindings[0]['bound_artifact_fqdn']==graph.nodes[0].matched_primitive,'Revision binding differs')
    binding=bindings[0]
    providers=db.execute("SELECT a.artifact_id,v.version_id,a.fqdn,l.import_module,l.source_symbol FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN atoms l ON l.atom_id=a.artifact_id JOIN atom_versions lv ON lv.version_id=v.version_id WHERE a.fqdn=%s AND v.content_hash=%s AND lv.content_hash=v.content_hash AND a.status='approved' AND a.is_publishable AND l.status='approved' AND l.is_publishable AND v.is_latest AND lv.is_latest AND v.trust_tier=3 AND lv.trust_tier=3",(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
    require(len(providers)==1,'Exact approved provider missing')
    provider=providers[0]
    require(provider['import_module']+'.'+provider['source_symbol']==graph.nodes[0].matched_primitive,'Runtime binding differs')
    require(binding['evidence_summary']==dict(provider_version_id=str(provider['version_id']),scope_audit_sha256=imported['scope_audit_sha256']),'Binding evidence differs')
    executed=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='period-frequency-execution.v1' AND passed",(REPLAY,)).fetchall()
    require(len(executed)==1 and sha(inspect.getfile(frequency_from_period))==executed[0]['details']['provider_source_sha256'],'Provider source changed')
    require(list(inspect.signature(frequency_from_period).parameters)==['period_seconds'],'Provider signature differs')
    for table,selected in [('artifact_io_specs',version),('artifact_io_specs',provider['version_id']),('atom_io_specs',provider['version_id'])]:
        columns=['direction','name','ordinal','type_desc','constraints','required','default_value_repr']
        if table=='artifact_io_specs':columns+=['dim_signature']
        ports=[]
        for direction,values in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
            for ordinal,port in enumerate(values):
                ports.append({key:direction if key=='direction' else ordinal if key=='ordinal' else getattr(port,key) for key in columns})
        require(db.execute(f'SELECT {",".join(columns)} FROM {table} WHERE version_id=%s ORDER BY direction,ordinal',(selected,)).fetchall()==ports,'Stored port contract differs')
    expected_deps=[]
    for source in [ORIGINAL,REPLAY,EXECUTION]:
        target=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(source,)).fetchone()
        expected_deps.append(dict(dependency_artifact_fqdn=target['fqdn'],dependency_content_hash=target['content_hash'],dependency_role='cdg',optional=False,port_name='',
            binding_metadata=dict(source_version_id=source,scope='Immutable source proof and qualified numerical provenance; runtime uses direct provider binding.')))
    deps=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional,port_name,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s',(version,)).fetchall()
    require(sorted(deps,key=lambda r:r['dependency_content_hash'])==sorted(expected_deps,key=lambda r:r['dependency_content_hash']),'Mandatory provenance differs')
    for selected in [version,str(provider['version_id']),EXECUTION]:
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(selected,)).fetchone(),'Failed version audit')
    served=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_id=%s',(ARTIFACT,)).fetchone()['n']
    require(served==int(approved),'Serving state differs')
    return graph,imported
