"""Check the qualified free_schrodinger parent and provider in the caller's transaction."""
import importlib
import inspect
import json
from pathlib import Path
from scripts.stage_remaining_legacy_physics_revisions import ROOT,sha,require
from sciona.physics_ingest.free_schrodinger_execution import build_free_schrodinger_execution,PRIMITIVE,SOURCE_VERSION,SOURCE_HASH
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg


def check_parent(db,qualification=None):
    graph=build_free_schrodinger_execution();digest,_,_=encode_execution_graph(graph)
    targets=db.execute("SELECT a.artifact_id,a.artifact_kind,a.fqdn,a.status,a.is_publishable,v.version_id,v.content_hash,v.trust_tier,v.is_latest,e.details FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e ON e.version_id=v.version_id WHERE e.runner_version='free_schrodinger-corrected-community.v1' AND e.passed AND e.status='completed' AND e.source_kind='automated'").fetchall()
    require(len(targets)==2 and {t['artifact_kind'] for t in targets}=={'atom','cdg'},'Unique approved free_schrodinger targets required')
    atom=next(t for t in targets if t['artifact_kind']=='atom');parent=next(t for t in targets if t['artifact_kind']=='cdg')
    retained=json.loads((ROOT/'docs/reviews/free_schrodinger_execution.json').read_text())
    for row in targets:
        require(row['status']=='approved' and row['is_publishable'] and row['is_latest'] and row['trust_tier']==3,'Free Schrodinger target not approved/latest Tier 3')
        require(db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(row['artifact_id'],)).fetchone(),'Free Schrodinger target not served')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(row['version_id'],)).fetchone(),'Failed free_schrodinger parent audit')
        details=row['details']
        require(details['publication_tier']==3 and details['source_parity_claim'] is False and details['source_version_id']==SOURCE_VERSION
            and details['source_content_hash']==SOURCE_HASH and details['execution_graph_sha256']==digest,'Free Schrodinger semantic identity differs')
        require(details['provider_version_id']==str(atom['version_id']) and details['provider_content_hash']==atom['content_hash'],'Free Schrodinger provider identity differs')
        require(details['corrected_proof']==retained['source_proof'],'Free Schrodinger corrected proof differs')
        bounds=db.execute('SELECT scope,bound_kind,validity_statement,evidence_ref_key,review_status,metadata '
            'FROM artifact_validity_bounds WHERE version_id=%s',(row['version_id'],)).fetchall()
        require(bounds==[dict(scope='version',bound_kind='regime',
            validity_statement='Positive constant mass/hbar, constant real three-component momentum and finite complex amplitude; fixed real x,y,z,t. Free nonrelativistic scalar plane wave, zero potential, zero momentum/amplitude supported. Compatible units required; no boundary or infinite-space normalized-state claim.',
            evidence_ref_key='openstax-free-schrodinger',review_status='automated_pass',
            metadata=dict(scope='Automated reviewed domain; caller establishes physical applicability.'))],
            'Free Schrodinger validity bounds differ')
        for name,value in details['evidence_sha256'].items():require(sha(ROOT/'docs/reviews'/name)==value,'Free Schrodinger approval evidence changed')
    for name,value in retained['implementation_sha256'].items():require(sha(ROOT/name)==value,'Free Schrodinger execution implementation changed')
    require(parent['content_hash']==digest,'Free Schrodinger parent hash differs')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
    require(_artifact_document_to_cdg(document,version_id=str(parent['version_id']),content_hash=digest,require_execution_envelope=True)==graph,'Free Schrodinger parent graph differs')
    provider=db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s',(atom['artifact_id'],)).fetchone()
    require(provider and provider['import_module']+'.'+provider['source_symbol']==PRIMITIVE,'Free Schrodinger provider import differs')
    require(db.execute('SELECT 1 FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.atom_id=%s AND v.version_id=%s AND v.content_hash=%s AND v.trust_tier=3 AND v.is_latest',(atom['artifact_id'],atom['version_id'],atom['content_hash'])).fetchone(),'Legacy free_schrodinger provider not served')
    module=importlib.import_module(provider['import_module']);fn=getattr(module,provider['source_symbol'])
    require(sha(Path(inspect.getfile(module)))==retained['provider_sha256'],'Free Schrodinger provider implementation changed')
    require(list(inspect.signature(fn).parameters)==[p.name for p in graph.nodes[0].inputs],'Free Schrodinger callable inputs differ')
    expected_binding=dict(node_id='wave',bound_artifact_fqdn=atom['fqdn'],bound_version_content_hash=atom['content_hash'],status='active',
        evidence_summary=dict(runtime_fqdn=PRIMITIVE,provider_version_id=str(atom['version_id']),output_aliases_by_ordinal=['wave_srepr','energy_srepr','gradient_json','laplacian_srepr','time_derivative_srepr','certificate_json']))
    require(db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s',(parent['version_id'],)).fetchall()==[expected_binding],'Free Schrodinger parent binding differs')
    for table,version in [('artifact_io_specs',parent['version_id']),('artifact_io_specs',atom['version_id']),('atom_io_specs',atom['version_id'])]:
        fields=['name','type_desc','constraints','required','default_value_repr']+(['dim_signature'] if table=='artifact_io_specs' else [])
        wanted=[dict(direction=direction,ordinal=i,**{k:getattr(p,k) for k in fields}) for direction,ports in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)] for i,p in enumerate(ports)]
        require(db.execute('SELECT direction,ordinal,'+','.join(fields)+' FROM '+table+' WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()==wanted,'Free Schrodinger parent ports differ')
    source=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
    require(source and source['content_hash']==SOURCE_HASH,'Free Schrodinger original source changed')
    require(db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s',(parent['version_id'],)).fetchall()==[dict(dependency_artifact_fqdn=source['fqdn'],dependency_content_hash=SOURCE_HASH,dependency_role='cdg',optional=False)],'Free Schrodinger parent provenance differs')
    require(not db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(atom['version_id'],)).fetchone(),'Unexpected free_schrodinger provider dependency')
    record=dict(version_id=str(parent['version_id']),artifact_id=str(parent['artifact_id']),graph_sha256=digest,provider_version_id=str(atom['version_id']),provider_content_hash=atom['content_hash'])
    if qualification is not None:require(record==qualification,'Qualified free_schrodinger parent changed')
    return graph,record
