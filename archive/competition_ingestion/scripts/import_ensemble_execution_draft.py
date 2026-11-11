#!/usr/bin/env python3
"""Store and bind the exact full ensemble graph without approving it."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.import_ensemble_adapter_drafts import verify_local_evidence
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_residual_execution_drafts import ensure_row
from scripts.import_riemannian_execution import SOURCE_ID, SOURCE_VERSION, SOURCE_HASH


def intake(root, reference_dir, apply=False):
    verify_local_evidence(root, reference_dir)
    graph = build_ensemble_execution_graph()
    wanted = {node.matched_primitive for node in graph.nodes}
    providers = {}
    for name in ['sciona-atoms', 'sciona-atoms-signal', 'sciona-atoms-ml']:
        directory = root.parent/name
        for spec in _parse_registered_atoms(repo=ProviderRepo(name, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in wanted:
                if runtime in providers:
                    raise ValueError('Duplicate parsed runtime identity')
                providers[runtime] = dict(catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
                    content_hash=spec.content_hash, provider_sha256=hashlib.sha256(spec.file_path.read_bytes()).hexdigest())
    if set(providers) != wanted:
        raise ValueError('Incomplete provider identities')
    digest, nodes, edges = encode_execution_graph(graph)
    artifact_id = uuid5(SOURCE_ID, 'full-ensemble-execution.v1')
    version_id = uuid5(artifact_id, digest)
    fqdn = 'cdg.competition.solution.kaggle.barachant_seizure_1st.full_ensemble_execution'
    evidence_hashes = {name: hashlib.sha256((root/'docs/reviews'/name).read_bytes()).hexdigest()
        for name in ['ensemble_graph_structure.json', 'ensemble_witness_verification.json', 'ensemble_alignment_parity.json']}
    created = 0
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('ensemble-execution-draft-intake.v1'))")
        source = db.execute("SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id) WHERE a.artifact_id=%s AND v.version_id=%s AND e.runner_version='competition-intake.v1' AND e.passed FOR SHARE OF a,v,e", (SOURCE_ID, SOURCE_VERSION)).fetchone()
        if not source or source['content_hash'] != SOURCE_HASH:
            raise ValueError('Pinned conceptual source intake differs')
        created += ensure_row(db, 'artifacts', {'artifact_id': artifact_id}, dict(
            artifact_id=artifact_id, artifact_kind='cdg', fqdn=fqdn, status='draft', is_publishable=False,
            verified_leaf_coverage=0., leaf_count=len(nodes),
            description='Full eleven-model ensemble across three explicitly ordered family populations with identity-aligned rank blending. Draft composition; complete raw execution and promotion review pending.'))
        created += ensure_row(db, 'artifact_versions', {'version_id': version_id}, dict(
            version_id=version_id, artifact_id=artifact_id, content_hash=digest,
            semver='0.0.0+execution.'+digest[:12], is_latest=False, trust_tier=3))
        for table, items, fields, keys in [
            ('artifact_cdg_nodes', nodes, ['node_id', 'parent_node_id', 'name', 'description', 'concept_type', 'status', 'matched_primitive', 'type_signature'], ['node_id']),
            ('artifact_cdg_edges', edges, ['source_id', 'target_id', 'output_name', 'input_name'], ['source_id', 'target_id', 'output_name', 'input_name']),
        ]:
            for item in items:
                row = {'version_id': version_id, **{field: item[field] for field in fields}}
                created += ensure_row(db, table, {key: row[key] for key in ['version_id', *keys]}, row)
        consumed = {(edge.target_id, edge.input_name) for edge in graph.edges}
        producers = {edge.source_id for edge in graph.edges}
        for direction, ports in [
            ('input', [port for node in graph.nodes for port in node.inputs if (node.node_id, port.name) not in consumed]),
            ('output', [port for node in graph.nodes if node.node_id not in producers for port in node.outputs]),
        ]:
            if len({port.name for port in ports}) != len(ports):
                raise ValueError('Ambiguous graph boundary')
            for ordinal, port in enumerate(ports):
                row = dict(artifact_id=artifact_id, version_id=version_id, direction=direction, name=port.name,
                    ordinal=ordinal, type_desc=port.type_desc, constraints=port.constraints,
                    required=port.required, default_value_repr=port.default_value_repr)
                created += ensure_row(db, 'artifact_io_specs', {key: row[key] for key in ['artifact_id', 'version_id', 'direction', 'name']}, row)
        for node in graph.nodes:
            record = providers[node.matched_primitive]
            provider = db.execute("SELECT a.artifact_id,v.content_hash,e.details FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id) WHERE a.fqdn=%s AND v.version_id=%s AND v.is_latest AND e.runner_version=ANY(%s) AND e.passed FOR SHARE OF a,v,e", (record['catalog_fqdn'], record['version_id'], ['riemannian-provider-intake.v1', 'relative-power-provider-intake.v1', 'combined-feature-provider-intake.v1', 'feng-knn-provider-intake.v1', 'feng-xgb-provider-intake.v1', 'feng-expanded-provider-intake.v1', 'andriy-provider-intake.v1', 'ensemble-adapter-intake.v1'])).fetchall()
            if len(provider) != 1:
                raise ValueError('Unique provider intake required')
            provider = provider[0]
            if provider['content_hash'] != record['content_hash'] or provider['details']['source_sha256'] != record['provider_sha256']:
                raise ValueError('Exact provider intake missing or changed')
            actual_ports = db.execute('SELECT direction,name,ordinal,type_desc,constraints,required,default_value_repr FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (record['version_id'],)).fetchall()
            if actual_ports != sorted(provider['details']['interfaces'], key=lambda p: (p['direction'],p['ordinal'])):
                raise ValueError('Provider interface evidence changed')
            row = dict(version_id=version_id, node_id=node.node_id, bound_artifact_fqdn=record['catalog_fqdn'],
                bound_version_content_hash=record['content_hash'], binding_confidence=1., binding_source='ensemble-provider-binding.v1',
                status='active', alternatives=Jsonb([]), evidence_summary=Jsonb(dict(runtime_fqdn=node.matched_primitive,
                    provider_version_id=record['version_id'], output_aliases_by_ordinal=[p.name for p in node.outputs],
                    scope='Exact identity binding within nonpublishable draft; no runtime approval.')))
            created += ensure_row(db, 'artifact_cdg_bindings', {'version_id': version_id, 'node_id': node.node_id}, row)
        dependency = dict(dependent_version_id=version_id, dependency_artifact_fqdn=source['fqdn'], dependency_content_hash=SOURCE_HASH, port_name='')
        created += ensure_row(db, 'artifact_dependencies', dependency, {**dependency, 'dependency_role': 'cdg', 'optional': False,
            'binding_metadata': Jsonb(dict(scope='Mandatory conceptual source provenance, not an invoked runtime dependency.'))})
        evidence_id = uuid5(version_id, 'ensemble-execution-draft-intake.v1')
        created += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, dict(
            evidence_id=evidence_id, artifact_id=artifact_id, version_id=version_id, audit_type='asset_integrity_check',
            passed=True, status='completed', source_kind='automated', runner_version='ensemble-execution-draft-intake.v1',
            details=Jsonb(dict(execution_graph=graph.model_dump(mode='json'), reference_report_sha256=evidence_hashes,
                scope='Draft structure, identity bindings and serialization integrity only. Full raw runtime, provider semantic reviews and publication approval remain separate.'))))
        document = db.execute('SELECT get_artifact_document(%s) AS d', (fqdn,)).fetchone()['d']
        if _artifact_document_to_cdg(document, version_id=str(version_id), content_hash=digest, require_execution_envelope=True) != graph:
            raise ValueError('Catalog graph roundtrip differs')
        if db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (artifact_id,)).fetchone():
            raise ValueError('Draft graph unexpectedly served')
        if not apply:
            db.rollback()
    return dict(applied=apply, rows_created=created, nodes=len(nodes), edges=len(edges), bindings=len(graph.nodes),
        catalog_roundtrip='passed', publication='draft')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(intake(Path(__file__).resolve().parents[1], args.reference_dir, args.apply), indent=2))
