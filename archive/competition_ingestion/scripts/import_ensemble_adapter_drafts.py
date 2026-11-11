#!/usr/bin/env python3
"""Intake exact ensemble adapter drafts without approving publication."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms, AtomSeedRow
from scripts.inventory_ensemble_adapters import inventory
from scripts.import_residual_execution_drafts import ensure_row


def verify_local_evidence(root, reference_dir):
    provider = root.parent/'sciona-atoms-signal/src/sciona/atoms/riemannian_bci/signal_processing'
    def check(path, sha):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Reviewed source/evidence drift: '+path.name)
    structure = json.loads((root/'docs/reviews/ensemble_graph_structure.json').read_text())
    if structure['graph_digest'] != inventory(root)['graph_digest']:
        raise ValueError('Reviewed graph differs')
    for name, sha in structure['files_sha256'].items():
        path = root/'sciona'/name if name == 'ensemble_execution.py' else root/'tests'/name if name.startswith('test_') else provider/name
        check(path, sha)
    alignment = json.loads((root/'docs/reviews/ensemble_alignment_parity.json').read_text())
    check(reference_dir/'make_blend.py', alignment['source_sha256'])
    check(provider/'ensemble_alignment.py', alignment['provider_sha256'])
    check(root/'scripts/validate_ensemble_alignment.py', alignment['validator_sha256'])
    check(root/'tests/test_ensemble_alignment.py', alignment['test_sha256'])
    if len(alignment['cases']) != 3 or not all(case['exact'] for case in alignment['cases']):
        raise ValueError('Final blend source cases incomplete')
    witness = json.loads((root/'docs/reviews/ensemble_witness_verification.json').read_text())
    check(root/'tests/test_ensemble_witnesses.py', witness['test_sha256'])
    if not witness['actual_registered_witnesses'] or witness['nodes_simulated'] != 155:
        raise ValueError('Full graph witness evidence missing')


def intake(root, reference_dir, apply=False):
    verify_local_evidence(root, reference_dir)
    closure = inventory(root)
    records = {record['runtime_fqdn']: record for record in closure['providers']}
    specs = {}
    for name in ['sciona-atoms-signal']:
        directory = root.parent/name
        for spec in _parse_registered_atoms(repo=ProviderRepo(name, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in records:
                if runtime in specs:
                    raise ValueError('Duplicate runtime identity')
                specs[runtime] = spec
    if set(specs) != set(records):
        raise ValueError('Parsed provider closure changed')
    references = {name: hashlib.sha256((root/'docs/reviews'/name).read_bytes()).hexdigest()
                  for name in ['ensemble_alignment_parity.json', 'ensemble_graph_structure.json', 'ensemble_witness_verification.json']}
    counts = Counter()
    identities = {}
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('ensemble-adapter-intake.v1'))")
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        for runtime, spec in sorted(specs.items()):
            record = records[runtime]
            if (str(spec.version_id) != record['version_id'] or spec.content_hash != record['content_hash']
                    or hashlib.sha256(spec.file_path.read_bytes()).hexdigest() != record['provider_sha256']):
                raise ValueError('Provider version drift during intake')
            atom_id = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)
            existing = db.execute('SELECT atom_id,status,is_publishable FROM atoms WHERE fqdn=%s FOR UPDATE', (spec.fqdn,)).fetchone()
            if existing and (existing['atom_id'] != atom_id or existing['status'] != 'flagged' or existing['is_publishable']):
                raise ValueError('Existing identity or publication state differs; separate review required')
            owner = db.execute('SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name=%s', (spec.repo_name,)).fetchall()
            if len(owner) != 1:
                raise ValueError('Unique provider repository ownership required')
            fields = {name: getattr(spec, name) for name in ['fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi']}
            fields.update(source_package=spec.namespace_root, status='flagged', is_publishable=False)
            atom = AtomSeedRow(**fields).as_dict(owner_id=str(owner[0]['owner_id']), source_repo_id=str(owner[0]['source_repo_id']))
            atom['atom_id'] = atom_id
            counts['legacy_provider_drafts'] += ensure_row(db, 'atoms', {'atom_id': atom_id}, atom)
            canonical = {key: value for key, value in atom.items() if key in columns}
            canonical.update(artifact_id=atom_id, artifact_kind='atom', status='draft', is_publishable=False)
            counts['canonical_provider_drafts'] += ensure_row(db, 'artifacts', {'artifact_id': atom_id}, canonical)
            for table, key in [('atom_versions', 'atom_id'), ('artifact_versions', 'artifact_id')]:
                counts['versions'] += ensure_row(db, table, {'version_id': spec.version_id}, dict(
                    version_id=spec.version_id, **{key: atom_id}, content_hash=spec.content_hash,
                    semver=spec.semver, is_latest=True, trust_tier=3, s3_key='', fingerprint=spec.fingerprint))
            for table, key in [('atom_io_specs', 'atom_id'), ('artifact_io_specs', 'artifact_id')]:
                for port in record['interfaces']:
                    row = {key: atom_id, 'version_id': spec.version_id, **port}
                    counts['interface_rows'] += ensure_row(db, table,
                        {name: row[name] for name in [key, 'version_id', 'direction', 'name']}, row)
            evidence_id = uuid5(UUID(str(spec.version_id)), 'ensemble-adapter-intake.v1')
            counts['integrity_evidence'] += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, dict(
                evidence_id=evidence_id, artifact_id=atom_id, version_id=spec.version_id,
                audit_type='asset_integrity_check', passed=True, status='completed', source_kind='automated',
                runner_version='ensemble-adapter-intake.v1', details=Jsonb(dict(
                    runtime_fqdn=runtime, source_sha256=record['provider_sha256'], interfaces=record['interfaces'],
                    reference_report_sha256=references, dependency_runtimes=record.get('dependencies', []),
                    scope='Exact adapter identity/interface intake only. Structural, witness and final-blend references retained. Full ensemble runtime and semantic promotion review remain separate; no publication approval.'))))
            identities[runtime] = atom_id
        for runtime, record in records.items():
            for dependency in record.get('dependencies', []):
                source, target = specs[runtime], specs[dependency]
                key = dict(dependent_version_id=source.version_id, dependency_artifact_fqdn=target.fqdn,
                           dependency_content_hash=target.content_hash, port_name='')
                counts['dependencies'] += ensure_row(db, 'artifact_dependencies', key, {**key,
                    'dependency_role': 'logic_atom', 'optional': False,
                    'binding_metadata': Jsonb(dict(runtime_fqdn=dependency, provider_version_id=str(target.version_id),
                        scope='Direct registered runtime dependency from reviewed static closure; publication pending.'))})
        served = db.execute('SELECT count(*) AS n FROM catalog_atoms_served WHERE atom_id=ANY(%s)', (list(identities.values()),)).fetchone()['n']
        if served:
            raise ValueError('Draft provider unexpectedly served')
        if not apply:
            db.rollback()
    return dict(applied=apply, counts=dict(counts), providers=len(records), served_providers=0,
        publication='pending_review')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(intake(Path(__file__).resolve().parents[1], args.reference_dir, args.apply), indent=2))
