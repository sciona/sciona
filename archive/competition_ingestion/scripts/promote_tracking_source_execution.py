#!/usr/bin/env python3
"""Atomically intake and approve the reviewed original analytical tracking realization at Tier 3."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID,uuid5,NAMESPACE_URL
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms,AtomSeedRow
from sciona.tracking_source_execution import build_tracking_source_execution,PRIMITIVE,SOURCE_VERSION,SOURCE_HASH
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.cdg_projection import build_published_cdg_projection
from scripts.import_residual_execution_drafts import ensure_row
from scripts.validate_tracking_source_execution import validate

REFERENCE='edwinst-tracking-source-f1a6e6396916'
URL='https://github.com/edwinst/trackml_solution/tree/f1a6e63969167159ef72c4ba32897e5aef7c3888'
LIMITATIONS=[
    'Automated Tier 3 only; no human-reviewed Tier 1 or Tier 2 usage claim.',
    'Original analytical observation-only flow; no cell features, learned calibration, scoring, tuning or nonphysical postprocessing.',
    'Finite real detector/observation matrices with six explicit source-API columns, integral bounded identifiers, positive module half lengths.',
    'Source-compatible cylinders and seven-ring, three-gap caps with occupied neighbor layers required; unsupported geometry may fail.',
    'Observation coordinates round to finite float32 before original float64 source arithmetic; labels align with input row order.',
    'Small-event ranking and early redundant-track pruning are explicit option changes; no exact full-competition configuration parity claim.',
    'Synthetic exact helix recovery and state isolation verified; no actual-event accuracy or throughput claim.',
    'Provider source closure, licenses, loader and compatibility implementation must match reviewed hashes.',
]



async def promote(root,apply=False):
    report=await validate(root)
    retained=json.loads((root/'docs/reviews/tracking_source_execution.json').read_text())
    if report!=retained:raise ValueError('Retained execution evidence differs from fresh validation')
    tests=json.loads((root/'docs/reviews/tracking_source_test_review.json').read_text())
    if tests['test_results']!=dict(tests=35,failures=0,errors=0,skipped=0):raise ValueError('Complete tests required')
    for path,sha in tests['test_source_sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=sha:raise ValueError('Test source changed')
    directory=root.parent/'sciona-atoms-physics'
    specs=[s for s in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-physics',directory),artifact_root=directory/'src/sciona/atoms') if s.import_module+'.'+s.source_symbol==PRIMITIVE]
    if len(specs)!=1:raise ValueError('Unique provider identity required')
    spec=specs[0]
    if hashlib.sha256(spec.file_path.read_bytes()).hexdigest()!=report['provider_closure_sha256']['atoms.py'] or tests['provider_sha256']!=report['provider_closure_sha256']['atoms.py']:raise ValueError('Provider source differs')
    graph=build_tracking_source_execution();digest,nodes,edges=encode_execution_graph(graph)
    if report['graph_digest']!=digest or report['full_runner_cases']!=3 or report['recovered_helices_per_case']!=8 or not report['exact_membership']:raise ValueError('Execution gates differ')
    evidence_hashes={p:hashlib.sha256((root/'docs/reviews'/p).read_bytes()).hexdigest() for p in ['tracking_source_execution.json','tracking_source_test_review.json','tracking_source_review.md']}
    atom_id=uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)
    rollup=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass',developer_semantics_status='pass',review_status='approved',review_semantic_verdict='pass',review_developer_semantics_verdict='pass',trust_readiness='ready',review_limitations=LIMITATIONS,review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits',parity_coverage_level='positive_and_negative',parity_test_status='pass')
    created=0
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('tracking-source-community.v1'))")
        source=db.execute('SELECT a.artifact_id,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s FOR SHARE OF a,v',(SOURCE_VERSION,)).fetchone()
        if not source or source['content_hash']!=SOURCE_HASH:raise ValueError('Exact source version missing')
        graph_id=uuid5(source['artifact_id'],'tracking-source-execution.v1');version_id=uuid5(graph_id,digest);fqdn=source['fqdn']+'.tracking_source_execution'
        owners=db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-physics'").fetchall()
        if len(owners)!=1:raise ValueError('Ambiguous owner')
        fields={name:getattr(spec,name) for name in ['fqdn','namespace_root','namespace_path','repo_name','source_module_path','import_module','source_symbol','description','domain_tags','source_kind','is_ffi']}
        fields.update(source_package=spec.namespace_root,status='flagged',is_publishable=False)
        atom=AtomSeedRow(**fields).as_dict(owner_id=str(owners[0]['owner_id']),source_repo_id=str(owners[0]['source_repo_id']));atom['atom_id']=atom_id
        existing=db.execute('SELECT status,is_publishable FROM atoms WHERE atom_id=%s FOR UPDATE',(atom_id,)).fetchone()
        if existing:
            if existing['status']!='approved' or not existing['is_publishable']:raise ValueError('Unexpected existing provider state')
            atom.update(existing)
        created+=ensure_row(db,'atoms',{'atom_id':atom_id},atom)
        columns={r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        canonical={k:v for k,v in atom.items() if k in columns and k not in {'created_at','updated_at'}}
        canonical.update(artifact_id=atom_id,artifact_kind='atom',status='approved' if existing else 'draft')
        created+=ensure_row(db,'artifacts',{'artifact_id':atom_id},canonical)
        for table,key in [('atom_versions','atom_id'),('artifact_versions','artifact_id')]:
            created+=ensure_row(db,table,{'version_id':spec.version_id},dict(version_id=spec.version_id,**{key:atom_id},content_hash=spec.content_hash,semver=spec.semver,is_latest=True,trust_tier=3,s3_key='',fingerprint=spec.fingerprint))
        description='Complete original analytical tracking flow with hash-verified source closure and isolated invocation state. Automated Tier 3 for source-compatible geometry; synthetic recovery validation only.'
        existing_graph=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',(graph_id,)).fetchone()
        if existing_graph and existing_graph!=dict(status='approved',is_publishable=True):raise ValueError('Unexpected existing graph state')
        created+=ensure_row(db,'artifacts',{'artifact_id':graph_id},dict(artifact_id=graph_id,artifact_kind='cdg',fqdn=fqdn,description=description,**(existing_graph or dict(status='draft',is_publishable=False))))
        created+=ensure_row(db,'artifact_versions',{'version_id':version_id},dict(version_id=version_id,artifact_id=graph_id,content_hash=digest,semver='0.0.0+execution.'+digest[:12],is_latest=bool(existing_graph),trust_tier=3))
        for node in nodes:
            created+=ensure_row(db,'artifact_cdg_nodes',{'version_id':version_id,'node_id':node['node_id']},{'version_id':version_id,**{k:node[k] for k in ['node_id','parent_node_id','name','description','concept_type','status','matched_primitive','type_signature']}})
        for direction,ports in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
            for ordinal,port in enumerate(ports):
                common=dict(direction=direction,name=port.name,ordinal=ordinal,type_desc=port.type_desc,constraints=port.constraints,required=port.required,default_value_repr=port.default_value_repr)
                for table,key,identity,selected in [('atom_io_specs','atom_id',atom_id,spec.version_id),('artifact_io_specs','artifact_id',atom_id,spec.version_id),('artifact_io_specs','artifact_id',graph_id,version_id)]:
                    row={key:identity,'version_id':selected,**common}
                    if table=='artifact_io_specs':row['dim_signature']=port.dim_signature
                    created+=ensure_row(db,table,{k:row[k] for k in [key,'version_id','direction','name']},row)
        created+=ensure_row(db,'artifact_cdg_bindings',{'version_id':version_id,'node_id':'tracking'},dict(version_id=version_id,node_id='tracking',bound_artifact_fqdn=spec.fqdn,bound_version_content_hash=spec.content_hash,binding_confidence=1.,binding_source='tracking-source-community.v1',status='active',alternatives=Jsonb([]),evidence_summary=Jsonb(dict(runtime_fqdn=PRIMITIVE,provider_version_id=str(spec.version_id),output_aliases_by_ordinal=['track_labels']))))
        dependency=dict(dependent_version_id=version_id,dependency_artifact_fqdn=source['fqdn'],dependency_content_hash=SOURCE_HASH,port_name='')
        created+=ensure_row(db,'artifact_dependencies',dependency,{**dependency,'dependency_role':'cdg','optional':False,'binding_metadata':Jsonb(dict(scope='Mandatory original competition-source provenance; full analytical execution is implemented by the bound provider. Optional cell/calibrated paths are outside this realization.'))})
        title='Original analytical tracking implementation at pinned commit'
        created+=ensure_row(db,'references_registry',{'ref_id':REFERENCE},dict(ref_id=REFERENCE,ref_type='repository',title=title,url=URL))
        for identity,selected,kind in [(atom_id,UUID(str(spec.version_id)),'provider'),(graph_id,version_id,'execution')]:
            if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(selected,)).fetchone():raise ValueError('Unresolved failed audit of publication target')
            tables=[('artifact_references','artifact_id'),('atom_references','atom_id')] if kind=='provider' else [('artifact_references','artifact_id')]
            for table,key in tables:
                created+=ensure_row(db,table,{key:identity,'ref_key':REFERENCE},{key:identity,'ref_id':REFERENCE,'ref_key':REFERENCE,'title':title,'url':URL,'source':'llm_extracted','verified':True,'confidence':'high','relevance_note':'Pinned original analytical tracking implementation; source hashes and licenses are retained in the reviewed provider closure.'})
            for table,key in ([('artifact_audit_rollups','artifact_id'),('atom_audit_rollups','atom_id')] if kind=='provider' else [('artifact_audit_rollups','artifact_id')]):
                created+=ensure_row(db,table,{key:identity},{key:identity,**rollup})
            evidence_id=uuid5(selected,'tracking-source-community.v1')
            created+=ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},dict(evidence_id=evidence_id,artifact_id=identity,version_id=selected,audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version='tracking-source-community.v1',details=Jsonb(dict(kind=kind,publication_tier=3,review_source='automated',evidence_sha256=evidence_hashes,provider_version_id=str(spec.version_id),provider_content_hash=spec.content_hash,execution_graph_sha256=digest,source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,provider_closure_sha256=report['provider_closure_sha256'],source_parity_claim=False,limitations=LIMITATIONS))))
            bound_id=uuid5(selected,'tracking-source-domain.v1')
            created+=ensure_row(db,'artifact_validity_bounds',{'bound_id':bound_id},dict(bound_id=bound_id,artifact_id=identity,version_id=selected,scope='version',bound_kind='regime',validity_statement='Source-compatible finite cylindrical/planar-cap geometry; explicit six-column detector and observation matrices. No cell features or learned calibration. See provider contract and review limitations for bounds.',evidence_ref_key=REFERENCE,review_status='automated_pass',metadata=Jsonb(dict(scope='Automated reviewed domain; caller establishes geometric applicability.'))))
        for selected in [spec.version_id,version_id]:
            if db.execute('SELECT count(*) AS n FROM artifact_io_specs WHERE version_id=%s',(selected,)).fetchone()['n']!=6:
                raise ValueError('Unexpected canonical ports')
        if db.execute('SELECT count(*) AS n FROM atom_io_specs WHERE version_id=%s',(spec.version_id,)).fetchone()['n']!=6:
            raise ValueError('Unexpected legacy ports')
        if db.execute('SELECT count(*) AS n FROM artifact_cdg_bindings WHERE version_id=%s',(version_id,)).fetchone()['n']!=1:
            raise ValueError('Unexpected graph bindings')
        if db.execute('SELECT count(*) AS n FROM artifact_dependencies WHERE dependent_version_id=%s',(version_id,)).fetchone()['n']!=1:
            raise ValueError('Unexpected graph dependencies')
        if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(spec.version_id,)).fetchone():
            raise ValueError('Unexpected provider dependencies')
        doc=db.execute('SELECT get_artifact_document(%s) AS d',(fqdn,)).fetchone()['d']
        if _artifact_document_to_cdg(doc,version_id=str(version_id),content_hash=digest,require_execution_envelope=True)!=graph:raise ValueError('Catalog graph differs')
        topo=build_published_cdg_projection(artifact={'artifact_id':str(graph_id),'fqdn':fqdn},version={'version_id':str(version_id),'content_hash':digest},cdg=graph).topo_hash
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s",(atom_id,))
        db.execute("UPDATE atoms SET status='approved',is_publishable=true WHERE atom_id=%s",(atom_id,))
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=1,top_level_input_arity=5,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(topo,graph_id))
        db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s',(version_id,graph_id))
        if not db.execute('SELECT 1 FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.atom_id=%s AND v.version_id=%s AND v.is_latest',(atom_id,spec.version_id)).fetchone():raise ValueError('Provider not served')
        if not db.execute('SELECT 1 FROM catalog_artifacts_served a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_id=%s AND v.version_id=%s AND v.is_latest',(graph_id,version_id)).fetchone():raise ValueError('Graph not served')
        if not apply:db.rollback()
    return dict(applied=apply,rows_created=created,trust_tier=3,served_provider_versions=1,served_execution_cdgs=1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(promote(Path(__file__).resolve().parents[1],args.apply))))
