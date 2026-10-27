"""Check the qualified escape_speed parent and provider in the caller's transaction."""
import importlib
import inspect
import json
from pathlib import Path
from scripts.stage_remaining_legacy_physics_revisions import ROOT,sha,require
from sciona.physics_ingest.escape_speed_execution import build_escape_speed_execution,PRIMITIVE,SOURCE_VERSION,SOURCE_HASH
from sciona.physics_ingest.infall_speed_execution import build_infall_speed_execution
from sciona.physics_ingest.infall_speed_proof import SOURCE_VERSION as INFALL_SOURCE_VERSION, SOURCE_HASH as INFALL_SOURCE_HASH
from scripts.promote_escape_speed_execution import provider_snapshot
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg


def check_parent(db,qualification=None):
    graph=build_escape_speed_execution();digest,_,_=encode_execution_graph(graph)
    targets=db.execute("SELECT a.artifact_id,a.artifact_kind,a.fqdn,a.status,a.is_publishable,v.version_id,v.content_hash,v.trust_tier,v.is_latest,e.details FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e ON e.version_id=v.version_id WHERE ((e.runner_version='escape_speed-corrected-community.v1' AND a.artifact_kind='cdg') OR (e.runner_version='infall_speed-corrected-community.v1' AND a.artifact_kind='atom')) AND e.passed AND e.status='completed' AND e.source_kind='automated'").fetchall()
    require(len(targets)==2 and {t['artifact_kind'] for t in targets}=={'atom','cdg'},'Unique approved escape_speed targets required')
    atom=next(t for t in targets if t['artifact_kind']=='atom');parent=next(t for t in targets if t['artifact_kind']=='cdg')
    retained=json.loads((ROOT/'docs/reviews/escape_speed_execution.json').read_text())
    infall=build_infall_speed_execution();infall_digest=encode_execution_graph(infall)[0]
    infall_retained=json.loads((ROOT/'docs/reviews/infall_speed_execution.json').read_text())
    for row in targets:
        is_provider=row['artifact_kind']=='atom'
        source_version=INFALL_SOURCE_VERSION if is_provider else SOURCE_VERSION
        source_hash=INFALL_SOURCE_HASH if is_provider else SOURCE_HASH
        expected_digest=infall_digest if is_provider else digest
        proof=infall_retained['source_proof'] if is_provider else retained['source_proof']
        require(row['status']=='approved' and row['is_publishable'] and row['is_latest'] and row['trust_tier']==3,'Escape speed target not approved/latest Tier 3')
        require(db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(row['artifact_id'],)).fetchone(),'Escape speed target not served')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(row['version_id'],)).fetchone(),'Failed escape_speed parent audit')
        details=row['details']
        require(details['publication_tier']==3 and details['source_parity_claim'] is False and details['source_version_id']==source_version
            and details['source_content_hash']==source_hash and details['execution_graph_sha256']==expected_digest,'Escape speed semantic identity differs')
        require(details['provider_version_id']==str(atom['version_id']) and details['provider_content_hash']==atom['content_hash'],'Escape speed provider identity differs')
        require(details['corrected_proof']==proof,'Escape speed corrected proof differs')
        bounds=db.execute('SELECT scope,bound_kind,validity_statement,evidence_ref_key,review_status,metadata '
            'FROM artifact_validity_bounds WHERE version_id=%s',(row['version_id'],)).fetchall()
        require(bounds==[dict(scope='version',bound_kind='regime',
            validity_statement=('Positive G, test mass, fixed central mass and radius; negligible test-mass backreaction in exterior spherical field. Inward radial zero-energy motion, asymptotic rest at infinity. No finite two-body, drag or relativistic claim.' if is_provider else 'Positive G, test mass, fixed central mass and radius; negligible test-mass backreaction in exterior spherical field. Positive escape threshold and required launch energy; negative velocity output is inward counterpart only. Asymptotic rest at infinity. No finite two-body, drag or relativistic claim.'),
            evidence_ref_key='openstax-infall-zero-energy' if is_provider else 'openstax-escape-zero-energy',review_status='automated_pass',
            metadata=dict(scope='Automated reviewed domain; caller establishes physical applicability.'))],
            'Escape speed validity bounds differ')
        for name,value in details['evidence_sha256'].items():require(sha(ROOT/'docs/reviews'/name)==value,'Escape speed approval evidence changed')
    for name,value in retained['implementation_sha256'].items():require(sha(ROOT/name)==value,'Escape speed execution implementation changed')
    require(parent['content_hash']==digest,'Escape speed parent hash differs')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
    require(_artifact_document_to_cdg(document,version_id=str(parent['version_id']),content_hash=digest,require_execution_envelope=True)==graph,'Escape speed parent graph differs')
    provider=db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s',(atom['artifact_id'],)).fetchone()
    require(provider and provider['import_module']+'.'+provider['source_symbol']==PRIMITIVE,'Escape speed provider import differs')
    require(db.execute('SELECT 1 FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.atom_id=%s AND v.version_id=%s AND v.content_hash=%s AND v.trust_tier=3 AND v.is_latest',(atom['artifact_id'],atom['version_id'],atom['content_hash'])).fetchone(),'Legacy escape_speed provider not served')
    module=importlib.import_module(provider['import_module']);fn=getattr(module,provider['source_symbol'])
    require(sha(Path(inspect.getfile(module)))==retained['provider_sha256'],'Escape speed provider implementation changed')
    require(list(inspect.signature(fn).parameters)==[p.name for p in graph.nodes[0].inputs],'Escape speed callable inputs differ')
    expected_binding=dict(node_id='escape',bound_artifact_fqdn=atom['fqdn'],bound_version_content_hash=atom['content_hash'],status='active',
        evidence_summary=dict(runtime_fqdn=PRIMITIVE,provider_version_id=str(atom['version_id']),output_aliases_by_ordinal=['escape_speed','inward_counterpart_velocity','required_launch_energy','potential_energy']))
    require(db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s',(parent['version_id'],)).fetchall()==[expected_binding],'Escape speed parent binding differs')
    for table,version in [('artifact_io_specs',parent['version_id']),('artifact_io_specs',atom['version_id']),('atom_io_specs',atom['version_id'])]:
        port_graph=graph if version==parent['version_id'] else infall
        fields=['name','type_desc','constraints','required','default_value_repr']+(['dim_signature'] if table=='artifact_io_specs' else [])
        wanted=[dict(direction=direction,ordinal=i,**{k:getattr(p,k) for k in fields}) for direction,ports in [('input',port_graph.nodes[0].inputs),('output',port_graph.nodes[0].outputs)] for i,p in enumerate(ports)]
        require(db.execute('SELECT direction,ordinal,'+','.join(fields)+' FROM '+table+' WHERE version_id=%s ORDER BY direction,ordinal',(version,)).fetchall()==wanted,'Escape speed parent ports differ')
    source=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
    require(source and source['content_hash']==SOURCE_HASH,'Escape speed original source changed')
    require(db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s',(parent['version_id'],)).fetchall()==[dict(dependency_artifact_fqdn=source['fqdn'],dependency_content_hash=SOURCE_HASH,dependency_role='cdg',optional=False)],'Escape speed parent provenance differs')
    require(not db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(atom['version_id'],)).fetchone(),'Unexpected escape_speed provider dependency')
    require(provider_snapshot(db,atom['artifact_id'])==parent['details']['reused_provider_snapshot_sha256'],'Reused provider records changed')
    require(str(atom['version_id'])==retained['reused_provider']['version_id'] and atom['content_hash']==retained['reused_provider']['content_hash'],'Pinned reused provider differs')
    record=dict(version_id=str(parent['version_id']),artifact_id=str(parent['artifact_id']),graph_sha256=digest,provider_version_id=str(atom['version_id']),provider_content_hash=atom['content_hash'])
    if qualification is not None:require(record==qualification,'Qualified escape_speed parent changed')
    return graph,record
