"""Transactionally stage the numerical residual graph; default is rollback.

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
from sciona.residual_classifier_graph import build_residual_classifier_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.import_residual_execution_drafts import ensure_row
from scripts.review_conditional_correction import SOURCE_VERSION, SOURCE_HASH, require
from scripts.plan_residual_classifier_draft import ROOT, audit, plan

RUNNER = 'residual-classifier-draft.v1'


def stage(apply=False):
    proposed, semantic = plan(), audit()
    graph = build_residual_classifier_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require(digest == proposed['graph_sha256'], 'Plan drift')
    directory = ROOT.parent / 'sciona-atoms-ml'
    specs = {s.fqdn: s for s in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
                                                       artifact_root=directory / 'src/sciona/atoms')}
    created = 0
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
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
        identity, version = UUID(proposed['artifact_id']), UUID(proposed['version_id'])
        ensure('artifacts', {'artifact_id': identity}, dict(artifact_id=identity, artifact_kind='cdg', fqdn=proposed['fqdn'],
            description=semantic['scope'], status='draft', is_publishable=False))
        ensure('artifact_versions', {'version_id': version}, dict(version_id=version, artifact_id=identity,
            content_hash=digest, semver='0.0.0+execution.' + digest[:12], is_latest=False, trust_tier=3))
        for table, rows, keys in [('artifact_cdg_nodes', nodes, ['node_id']),
                                 ('artifact_cdg_edges', edges, ['source_id', 'target_id', 'output_name', 'input_name'])]:
            for item in rows:
                row = dict(version_id=version, **item)
                ensure(table, {k: row[k] for k in ['version_id', *keys]}, row)
        ports(identity, version, proposed['boundary_inputs'], proposed['boundary_outputs'])
        for binding in proposed['bindings']:
            ensure('artifact_cdg_bindings', dict(version_id=version, node_id=binding['node_id']),
                dict(version_id=version, node_id=binding['node_id'], bound_artifact_fqdn=binding['fqdn'],
                     bound_version_content_hash=binding['content_hash'], binding_confidence=1., binding_source=RUNNER,
                     status='active', alternatives=Jsonb([]), evidence_summary=Jsonb(dict(runtime_fqdn=binding['runtime_fqdn'],
                     provider_version_id=binding['version_id'], output_aliases_by_ordinal=[p['name'] for p in binding['outputs']]))))
        provenance = proposed['mandatory_provenance']
        dependency = dict(dependent_version_id=version, dependency_artifact_fqdn=provenance['fqdn'],
                          dependency_content_hash=SOURCE_HASH, port_name='')
        ensure('artifact_dependencies', dependency, dict(**dependency, dependency_role='cdg', optional=False,
                                                        binding_metadata=Jsonb(dict(scope=provenance['scope']))))
        targets.append((identity, version))
        for target, selected in targets:
            evidence = uuid5(selected, RUNNER)
            ensure('artifact_audit_evidence', {'evidence_id': evidence}, dict(evidence_id=evidence, artifact_id=target,
                version_id=selected, audit_type='asset_integrity_check', passed=True, status='completed', source_kind='automated',
                runner_version=RUNNER, details=Jsonb(dict(scope='Reviewed draft import only; no publication approval', semantic_review=semantic))))
            require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (target,)).fetchone(),
                    'Draft unexpectedly served')
        document = db.execute('SELECT get_artifact_document(%s) AS d', (proposed['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document, version_id=str(version), content_hash=digest,
                    require_execution_envelope=True) == graph, 'Catalog serialization differs')
        require(audit() == semantic, 'Evidence changed during transaction')
        if not apply:
            db.rollback()
    return dict(applied=apply, rows_created=created, draft_atoms=len(proposed['atoms']), draft_cdgs=1, approved=False,
                catalog_graph_roundtrip=True, graph_sha256=digest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(stage(args.apply)))
