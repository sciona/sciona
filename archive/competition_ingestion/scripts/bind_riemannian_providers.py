#!/usr/bin/env python3
"""Intake exact Riemannian provider versions, interfaces and runtime dependencies."""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from uuid import UUID, uuid5, NAMESPACE_URL
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms, AtomSeedRow
from sciona.physics_ingest.pdg_evidence import _digest
from scripts.import_residual_execution_drafts import ensure_row
from scripts.import_riemannian_execution import validated_graph

WINDOW = 'sciona.atoms.riemannian_bci.signal_processing.segment_windows.window_signal_segments'
AUDIO = 'sciona.atoms.audio_speech.atoms.audio_windows'
PREPARE = 'sciona.atoms.riemannian_bci.signal_processing.source_inputs.prepare_segment_windows'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--component', choices=['riemannian', 'relative_power', 'combined_feature', 'feng_knn', 'feng_xgb', 'feng_expanded'], default='riemannian')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report, graph, digest, _, _ = validated_graph(root, args.reference_dir, args.component)
    prefix = args.component.replace('_', '-')
    feng = args.component in {'feng_knn', 'feng_xgb', 'feng_expanded'}
    feng_base = 'sciona.atoms.riemannian_bci.signal_processing.'
    feng_filter = feng_base + 'feng_filter.feng_resample_filter'
    feng_fft = feng_base + 'feng_fft.feng_fft_features'
    expanded = args.component == 'feng_expanded'
    if expanded:
        feng_fft = feng_base + 'feng_expanded_features.feng_expanded_features'
    desired = {n.matched_primitive for n in graph.nodes} | ({feng_filter, feng_fft} if feng else {WINDOW, AUDIO})
    parsed = {}
    for repo in ['sciona-atoms', 'sciona-atoms-signal', 'sciona-atoms-ml']:
        directory = root.parent / repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo, directory), artifact_root=directory / 'src/sciona/atoms'):
            runtime = spec.import_module + '.' + spec.source_symbol
            if runtime in desired:
                parsed.setdefault(runtime, []).append(spec)
    if set(parsed) != desired or any(len(v) != 1 for v in parsed.values()):
        raise ValueError('unique parsed identity required for complete provider closure')
    specs = {k: v[0] for k, v in parsed.items()}
    counts = Counter()
    with psycopg.connect(dotenv_values(root / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (prefix + '-provider-binding.v1',))
        intakes = db.execute('SELECT * FROM artifact_audit_evidence WHERE runner_version=%s AND passed FOR SHARE', (prefix + '-execution-intake.v1',)).fetchall()
        if len(intakes) != 1:
            raise ValueError('unique execution intake required')
        intake = intakes[0]
        if intake['details']['execution_report_sha256'] != _digest(report) or intake['details']['execution_graph'] != graph.model_dump(mode='json'):
            raise ValueError('catalog execution evidence differs')
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        identities = {}
        for runtime, spec in sorted(specs.items()):
            fn = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
            signature = inspect.signature(fn)
            sha = hashlib.sha256(spec.file_path.read_bytes()).hexdigest()
            expected = report['provider_hashes'].get(runtime, report['execution_module_hashes'].get(spec.import_module))
            if expected != sha or fn.__module__ + '.' + fn.__name__ != runtime:
                raise ValueError('validated provider identity or source differs')
            old = db.execute('SELECT * FROM atoms WHERE fqdn=%s FOR UPDATE', (spec.fqdn,)).fetchone()
            if old:
                if old['import_module'] != spec.import_module or old['source_symbol'] != spec.source_symbol:
                    raise ValueError('existing import identity differs')
                version = db.execute('SELECT * FROM atom_versions WHERE atom_id=%s AND is_latest FOR SHARE', (old['atom_id'],)).fetchone()
                if not version or str(version['version_id']) != str(spec.version_id) or version['content_hash'] != spec.content_hash:
                    raise ValueError('existing version differs; separate version review required')
                atom_id = old['atom_id']
            else:
                owner = db.execute('SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name=%s', (spec.repo_name,)).fetchall()
                if len(owner) != 1:
                    raise ValueError('unique provider ownership required')
                atom_id = uuid5(NAMESPACE_URL, 'sciona-provider-draft:' + spec.fqdn)
                fields = {name: getattr(spec, name) for name in ['fqdn', 'namespace_root', 'namespace_path', 'repo_name', 'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi']}
                fields.update(source_package=spec.namespace_root, status='flagged', is_publishable=False)
                old = AtomSeedRow(**fields).as_dict(owner_id=str(owner[0]['owner_id']), source_repo_id=str(owner[0]['source_repo_id']))
                old['atom_id'] = atom_id
                counts['new_provider_identities'] += ensure_row(db, 'atoms', {'atom_id': atom_id}, old)
                ensure_row(db, 'atom_versions', {'version_id': spec.version_id}, dict(version_id=spec.version_id, atom_id=atom_id,
                    content_hash=spec.content_hash, semver=spec.semver, is_latest=True, trust_tier=3, s3_key='', fingerprint=spec.fingerprint))
            identities[runtime] = atom_id
            if old['status'] == 'approved' and old['is_publishable']:
                if not db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s', (atom_id,)).fetchone():
                    raise ValueError('approved dependency is not served')
                if runtime != AUDIO:
                    approved = db.execute("SELECT 1 FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id) WHERE a.artifact_id=%s AND a.status='approved' AND a.is_publishable AND v.version_id=%s AND v.content_hash=%s AND v.is_latest AND e.passed AND e.runner_version IN ('riemannian-provider-community.v1','relative-power-provider-community.v1','combined-feature-provider-community.v1','feng-knn-provider-community.v1','feng-xgb-provider-community.v1','feng-expanded-provider-community.v1')", (atom_id, spec.version_id, spec.content_hash)).fetchone()
                    if not approved:
                        raise ValueError('reused canonical provider lacks exact approval')
                    for ordinal, (name, param) in enumerate(signature.parameters.items()):
                        port = db.execute("SELECT ordinal,required,default_value_repr FROM artifact_io_specs WHERE version_id=%s AND direction='input' AND name=%s", (spec.version_id, name)).fetchone()
                        required = param.default is inspect.Parameter.empty
                        if not port or port != dict(ordinal=ordinal, required=required, default_value_repr='' if required else repr(param.default)):
                            raise ValueError('reused provider interface differs')
                counts['existing_approved_dependencies_verified'] += 1
                continue
            if runtime == AUDIO:
                raise ValueError('existing audio dependency is not approved')
            canonical = {k: v for k, v in old.items() if k in columns and k not in {'created_at', 'updated_at'}}
            canonical.update(artifact_id=atom_id, artifact_kind='atom', status='draft', is_publishable=False)
            counts['canonical_identities_created'] += ensure_row(db, 'artifacts', {'artifact_id': atom_id}, canonical)
            counts['canonical_versions_created'] += ensure_row(db, 'artifact_versions', {'version_id': spec.version_id},
                dict(version_id=spec.version_id, artifact_id=atom_id, content_hash=spec.content_hash, semver=spec.semver,
                     is_latest=True, trust_tier=3, s3_key='', fingerprint=spec.fingerprint))
            matching = [n for n in graph.nodes if n.matched_primitive == runtime]
            ports = {p.name: p for n in matching for p in n.inputs}
            outputs = [(p.name, p.type_desc) for p in matching[0].outputs] if matching else [('windows', 'numpy.ndarray'), ('segment_indices', 'numpy.ndarray')]
            if feng and not matching:
                outputs = [('filtered_segment' if runtime == feng_filter else 'features', 'numpy.ndarray')]
            if spec.source_symbol == 'segment_probability_max':
                outputs = [('segment_probabilities', 'numpy.ndarray')]
            interfaces = []
            for ordinal, (name, param) in enumerate(signature.parameters.items()):
                required = param.default is inspect.Parameter.empty
                kind = ports[name].type_desc if name in ports else ('list' if name == 'segments' else 'int')
                if feng and name in {'segment', 'filtered_segment'}:
                    kind = 'numpy.ndarray'
                interfaces.append(dict(direction='input', name=name, ordinal=ordinal, type_desc=kind,
                    constraints='See exact version description for numerical/domain requirements.', required=required,
                    default_value_repr='' if required else repr(param.default)))
            for ordinal, (name, kind) in enumerate(outputs):
                interfaces.append(dict(direction='output', name=name, ordinal=ordinal, type_desc=kind,
                    constraints='Returned tuple order and shape follow the exact callable contract.', required=True, default_value_repr=''))
            for table, key in [('artifact_io_specs', 'artifact_id'), ('atom_io_specs', 'atom_id')]:
                for port in interfaces:
                    row = {key: atom_id, 'version_id': spec.version_id, **port}
                    counts['interface_ports_created'] += ensure_row(db, table, {k: row[k] for k in [key, 'version_id', 'direction', 'name']}, row)
            evidence_id = uuid5(UUID(str(spec.version_id)), prefix + '-provider-intake.v1')
            counts['provider_evidence_created'] += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, {
                'evidence_id': evidence_id, 'artifact_id': atom_id, 'version_id': spec.version_id, 'audit_type': 'asset_integrity_check',
                'passed': True, 'status': 'completed', 'source_kind': 'automated', 'runner_version': prefix + '-provider-intake.v1',
                'details': Jsonb({'runtime_fqdn': runtime, 'catalog_fqdn': spec.fqdn, 'source_sha256': sha,
                    'signature': str(signature), 'interfaces': interfaces, 'execution_evidence_id': str(intake['evidence_id']),
                    'execution_evidence_sha256': _digest(intake['details']), 'scope': 'Exact identity, complete interfaces/defaults and execution linkage; publication approval pending.'})})
        for node in graph.nodes:
            spec = specs[node.matched_primitive]
            binding = dict(version_id=intake['version_id'], node_id=node.node_id, bound_artifact_fqdn=spec.fqdn,
                bound_version_content_hash=spec.content_hash, binding_confidence=1., binding_source=prefix + '-provider-binding.v1',
                status='active', alternatives=Jsonb([]), evidence_summary=Jsonb({'runtime_fqdn': node.matched_primitive,
                    'provider_version_id': str(spec.version_id), 'output_aliases_by_ordinal': [p.name for p in node.outputs],
                    'identity_basis': 'unique parsed import_module/source_symbol and exact installed version'}))
            counts['bindings_created'] += ensure_row(db, 'artifact_cdg_bindings', {'version_id': intake['version_id'], 'node_id': node.node_id}, binding)
        runtime_dependencies = [(feng_base + 'feng_partitions.feng_filter_partitions', feng_filter), (feng_base + 'feng_partitions.feng_feature_partitions', feng_fft)] if feng else [(PREPARE, WINDOW), (WINDOW, AUDIO)]
        if expanded:
            runtime_dependencies[1] = (feng_base + 'feng_expanded_partitions.feng_expanded_partitions', feng_fft)
        for caller, dependency in runtime_dependencies:
            src, dst = specs[caller], specs[dependency]
            key = dict(dependent_version_id=src.version_id, dependency_artifact_fqdn=dst.fqdn,
                       dependency_content_hash=dst.content_hash, port_name='')
            counts['runtime_dependencies_created'] += ensure_row(db, 'artifact_dependencies', key, {**key,
                'dependency_role': 'logic_atom', 'optional': False, 'binding_metadata': Jsonb({'runtime_fqdn': dependency,
                    'provider_version_id': str(dst.version_id), 'scope': 'direct invoked registered Feng numerical dependency' if feng else 'direct invoked registered windowing dependency'})})
        if not args.apply:
            db.rollback()
    print(json.dumps({'applied': args.apply, 'counts': dict(counts), 'publication': 'pending_review'}, indent=2))


if __name__ == '__main__':
    main()
