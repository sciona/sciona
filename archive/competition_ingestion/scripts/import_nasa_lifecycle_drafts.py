"""Transactionally stage the decomposed domain graph; default is rollback.

This creates non-publishable drafts only. It does not waive outstanding license,
negative publication, served execution, or original domain-workflow gates.
"""
import argparse
import json
from uuid import UUID, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from scripts.plan_nasa_lifecycle_drafts import BUILDERS
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.import_residual_execution_drafts import ensure_row
from scripts.review_conditional_correction import SOURCE_VERSION, SOURCE_HASH, require
from scripts.plan_nasa_lifecycle_drafts import ROOT, audit, plan, check_parent

RUNNER = 'nasa-lifecycle-draft.v1'


def stage(apply=False):
    proposed, semantic = plan(), audit()
    directory = ROOT.parent / 'sciona-atoms-ml'
    specs = {s.fqdn: s for s in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
                                                       artifact_root=directory / 'src/sciona/atoms')}
    created = 0
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('residual-classifier-draft.v1'))")
        db.execute("SELECT pg_advisory_xact_lock(hashtext('nasa-domain-draft.v1'))")
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        shared_ids = [atom['artifact_id'] for atom in proposed['reused_atoms']]
        db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=ANY(%s::uuid[]) FOR SHARE', (shared_ids,)).fetchall()
        check_parent(db, proposed['parent_domain'])
        source = db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v '
            'USING(artifact_id) WHERE v.version_id=%s FOR SHARE OF a,v', (SOURCE_VERSION,)).fetchone()
        require(source and source['content_hash'] == SOURCE_HASH and source['status'] == 'draft' and
                not source['is_publishable'], 'Source intake changed')
        require(source['fqdn'] == proposed['mandatory_provenance']['fqdn'], 'Source provenance name differs')
        owners = db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r "
                            "USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        require(len(owners) == 1, 'Ambiguous owner')
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns "
                                                       "WHERE table_schema='public' AND table_name='artifacts'")}

        def ensure(table, identity, row):
            nonlocal created
            created += ensure_row(db, table, identity, row)

        def ports(identity, version, inputs, outputs, legacy=False):
            for direction, selected in [('input', inputs), ('output', outputs)]:
                for ordinal, port in enumerate(selected):
                    common = dict(version_id=version, direction=direction, ordinal=ordinal,
                                  **{k: port[k] for k in ['name', 'type_desc', 'constraints', 'required', 'default_value_repr']})
                    for table, key in [('artifact_io_specs', 'artifact_id')] + ([('atom_io_specs', 'atom_id')] if legacy else []):
                        row = {key: identity, **common}
                        if key == 'artifact_id':
                            row['dim_signature'] = port['dim_signature']
                        ensure(table, {k: row[k] for k in [key, 'version_id', 'direction', 'name']}, row)

        targets = []
        for binding in proposed['atoms']:
            spec = specs[binding['fqdn']]
            require(str(spec.version_id) == binding['version_id'] and spec.content_hash == binding['content_hash'], 'Provider changed')
            identity, version = UUID(binding['artifact_id']), UUID(binding['version_id'])
            fields = {k: getattr(spec, k) for k in ['fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi']}
            fields.update(source_package=spec.namespace_root, status='flagged', is_publishable=False)
            atom = AtomSeedRow(**fields).as_dict(owner_id=str(owners[0]['owner_id']), source_repo_id=str(owners[0]['source_repo_id']))
            atom['atom_id'] = identity
            ensure('atoms', {'atom_id': identity}, atom)
            canonical = {k: v for k, v in atom.items() if k in columns and k not in {'created_at', 'updated_at'}}
            canonical.update(artifact_id=identity, artifact_kind='atom', status='draft')
            ensure('artifacts', {'artifact_id': identity}, canonical)
            for table, key in [('atom_versions', 'atom_id'), ('artifact_versions', 'artifact_id')]:
                ensure(table, {'version_id': version}, dict(version_id=version, **{key: identity}, content_hash=spec.content_hash,
                    semver=spec.semver, is_latest=True, trust_tier=3, s3_key='', fingerprint=spec.fingerprint))
            ports(identity, version, binding['inputs'], binding['outputs'], legacy=True)
            targets.append((identity, version))
        for kind, selected_graph in proposed['graphs'].items():
            graph=BUILDERS[kind]()
            digest,nodes,edges=encode_execution_graph(graph)
            require(digest==selected_graph['graph_sha256'],'Lifecycle graph drift')
            identity,version=UUID(selected_graph['artifact_id']),UUID(selected_graph['version_id'])
            ensure('artifacts',{'artifact_id':identity},dict(artifact_id=identity,artifact_kind='cdg',fqdn=selected_graph['fqdn'],
                description=selected_graph['scope'],status='draft',is_publishable=False))
            ensure('artifact_versions',{'version_id':version},dict(version_id=version,artifact_id=identity,
                content_hash=digest,semver='0.0.0+execution.'+digest[:12],is_latest=False,trust_tier=3))
            for table,rows,keys in [('artifact_cdg_nodes',nodes,['node_id']),
                                  ('artifact_cdg_edges',edges,['source_id','target_id','output_name','input_name'])]:
                for item in rows:
                    row=dict(version_id=version,**item)
                    ensure(table,{key:row[key] for key in ['version_id',*keys]},row)
            ports(identity,version,selected_graph['boundary_inputs'],selected_graph['boundary_outputs'])
            for binding in selected_graph['bindings']:
                row=dict(version_id=version,node_id=binding['node_id'],bound_artifact_fqdn=binding['fqdn'],
                    bound_version_content_hash=binding['content_hash'],binding_confidence=1.,binding_source=RUNNER,
                    status='active',alternatives=Jsonb([]),evidence_summary=Jsonb(dict(runtime_fqdn=binding['runtime_fqdn'],
                    provider_version_id=binding['version_id'],output_aliases_by_ordinal=[port['name'] for port in binding['outputs']])))
                ensure('artifact_cdg_bindings',dict(version_id=version,node_id=binding['node_id']),row)
            parent=proposed['parent_domain']
            for fqdn,content_hash,source_version,scope in [
                (proposed['mandatory_provenance']['fqdn'],SOURCE_HASH,SOURCE_VERSION,'Mandatory original competition intake provenance; lifecycle reconstruction only.'),
                (parent['fqdn'],parent['graph_sha256'],parent['version_id'],'Mandatory approved domain structure provenance; shared atom bindings are invoked directly.')]:
                dependency=dict(dependent_version_id=version,dependency_artifact_fqdn=fqdn,dependency_content_hash=content_hash,port_name='')
                ensure('artifact_dependencies',dependency,dict(**dependency,dependency_role='cdg',optional=False,
                    binding_metadata=Jsonb(dict(scope=scope,source_version_id=source_version))))
            document=db.execute('SELECT get_artifact_document(%s) AS d',(selected_graph['fqdn'],)).fetchone()['d']
            require(_artifact_document_to_cdg(document,version_id=str(version),content_hash=digest,
                require_execution_envelope=True)==graph,'Catalog lifecycle serialization differs')
            targets.append((identity,version))
        for target,version in targets:
            evidence=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',{'evidence_id':evidence},dict(evidence_id=evidence,artifact_id=target,
                version_id=version,audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',
                runner_version=RUNNER,details=Jsonb(dict(scope='Reviewed lifecycle draft import only; no publication approval',semantic_review=semantic))))
            require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(target,)).fetchone(),'Draft unexpectedly served')
        check_parent(db,proposed['parent_domain'])
        require(audit()==semantic,'Lifecycle evidence changed during transaction')
        if not apply:db.rollback()
    return dict(applied=apply,rows_created=created,draft_atoms=len(proposed['atoms']),draft_cdgs=len(proposed['graphs']),
        reused_approved_atoms=len(proposed['reused_atoms']),approved=False,catalog_graph_roundtrips=True,
        graph_sha256={kind:graph['graph_sha256'] for kind,graph in proposed['graphs'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(stage(args.apply)))
