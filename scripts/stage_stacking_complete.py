"""Stage the decomposed Otto revision and providers without approving either.

Default execution rolls back. --apply requires current transaction evidence.
Original source rows and latest selection must remain unchanged.
"""
import argparse
import json
from pathlib import Path
from uuid import UUID, NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.architect.models import IOSpec
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.stacking_complete_graph import build_complete_graph
from scripts.import_residual_execution_drafts import ensure_row
from scripts.promote_otto_execution import boundary_ports, check_hashes
from scripts.review_stacking_complete_scope import ROOT, review, require, sha
from scripts.stage_remaining_legacy_physics_revisions import history

RUNNER = 'otto-decomposed-original-draft.v1'


def stage(apply=False, fail_after_writes=None):
    require(not apply or fail_after_writes is None, 'Failure injection requires rollback mode')
    reviews = ROOT/'docs/reviews'
    scope = review()
    require(scope == json.loads((reviews/'competition_otto_decomposed_complete_scope.json').read_text()),
            'Retained scope differs from current qualification')
    gates = json.loads((reviews/'competition_otto_decomposed_review_gates.json').read_text())
    require(gates['status'] == 'offline_review_gates_passed' and len(gates['rejected_scenarios']) == 21
            and gates['pre_catalog_baseline_passed_before_and_after'], 'Current offline review gates required')
    check_hashes(ROOT, gates['sha256'])
    if apply:
        transaction = json.loads((reviews/'competition_otto_decomposed_draft_transaction.json').read_text())
        require(transaction['passed'] and transaction['rollback_verified']
                and len(transaction['injected_failures_rolled_back']) == 3
                and transaction['scope_sha256'] == sha(reviews/'competition_otto_decomposed_complete_scope.json')
                and transaction['stager_sha256'] == sha(__file__)
                and transaction['validator_sha256'] == sha(ROOT/'scripts/validate_stacking_complete_transaction.py'),
                'Current staging transaction qualification required')
    graph = build_complete_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require(digest == scope['graph_sha256'], 'Graph differs from scope')
    identity, version = UUID(scope['artifact_id']), UUID(scope['corrected_version_id'])
    directory = ROOT.parent/'sciona-atoms-ml'
    parsed = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
                                     artifact_root=directory/'src/sciona/atoms')
    specs = {s.fqdn:s for s in parsed if s.fqdn in scope['providers']}
    require(len(specs) == 27, 'Exact provider inventory required')
    by_runtime = {p['runtime_fqdn']:fqdn for fqdn,p in scope['providers'].items()}
    interfaces = {n.node_id:{direction:[IOSpec.model_validate(p) for p in ports]
                            for direction,ports in scope['providers'][by_runtime[n.matched_primitive]]['ports'].items()}
                  for n in graph.nodes}
    boundary = boundary_ports(graph, interfaces)
    created = 0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        source = db.execute('SELECT fqdn,status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',
                            (identity,)).fetchone()
        require(source and source['status']=='draft' and not source['is_publishable'], 'Original must remain draft')
        require(history(db, scope['original_version_id']) == scope['original_history_sha256'], 'Source history differs')
        owners = db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        require(len(owners)==1, 'Unique provider ownership required')
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}

        def ensure(table, key, row):
            nonlocal created
            created += int(ensure_row(db, table, key, row))
            if fail_after_writes is not None and created == fail_after_writes:
                raise RuntimeError('injected complete-stack staging failure')

        def ports(owner, selected, contract, legacy=False):
            tables = [('artifact_io_specs','artifact_id')]
            if legacy:
                tables.append(('atom_io_specs','atom_id'))
            for table, owner_key in tables:
                count = 0
                for direction, values in contract.items():
                    for ordinal, port in enumerate(values):
                        require(not any(getattr(port,k,None) for k in ('data_kind','time_basis','provenance')),
                                'Populated port metadata cannot be discarded')
                        row = dict(version_id=selected, direction=direction, ordinal=ordinal,
                                   **{owner_key:owner},
                                   **{k:getattr(port,k) for k in ('name','type_desc','constraints','required','default_value_repr')})
                        if table == 'artifact_io_specs':
                            row['dim_signature'] = port.dim_signature
                        ensure(table, {k:row[k] for k in (owner_key,'version_id','direction','name')}, row)
                        count += 1
                require(db.execute('SELECT count(*) AS n FROM '+table+' WHERE version_id=%s',
                                   (selected,)).fetchone()['n'] == count, 'Unexpected additional ports')

        provider_ids = []
        for fqdn, spec in sorted(specs.items()):
            qualified = scope['providers'][fqdn]
            require(str(spec.version_id)==qualified['version_id'] and spec.content_hash==qualified['content_hash']
                    and sha(spec.file_path)==qualified['source_sha256'], 'Parsed provider differs')
            owner = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+fqdn)
            provider_ids.append(owner)
            fields = {k:getattr(spec,k) for k in ('fqdn','namespace_root','namespace_path','repo_name',
                      'source_module_path','import_module','source_symbol','description','domain_tags','source_kind','is_ffi')}
            fields.update(source_package=spec.namespace_root, status='flagged', is_publishable=False)
            atom = AtomSeedRow(**fields).as_dict(owner_id=str(owners[0]['owner_id']), source_repo_id=str(owners[0]['source_repo_id']))
            atom['atom_id'] = owner
            ensure('atoms', dict(atom_id=owner), atom)
            canonical = {k:v for k,v in atom.items() if k in columns}
            canonical.update(artifact_id=owner, artifact_kind='atom', status='draft')
            ensure('artifacts', dict(artifact_id=owner), canonical)
            for table, owner_key in [('atom_versions','atom_id'),('artifact_versions','artifact_id')]:
                ensure(table, dict(version_id=spec.version_id), dict(version_id=spec.version_id, **{owner_key:owner},
                    content_hash=spec.content_hash, semver=spec.semver, is_latest=True, trust_tier=3,
                    s3_key='', fingerprint=spec.fingerprint))
                require(db.execute('SELECT count(*) AS n FROM '+table+' WHERE '+owner_key+'=%s AND is_latest',
                                   (owner,)).fetchone()['n']==1, 'Ambiguous provider latest selection')
            ports(owner, spec.version_id,
                  {d:[IOSpec.model_validate(p) for p in ps] for d,ps in qualified['ports'].items()}, True)
            evidence = uuid5(UUID(str(spec.version_id)), RUNNER)
            ensure('artifact_audit_evidence', dict(evidence_id=evidence), dict(evidence_id=evidence, artifact_id=owner,
                version_id=spec.version_id, audit_type='asset_integrity_check', passed=True, status='completed',
                source_kind='automated', runner_version=RUNNER, details=Jsonb(dict(
                    scope='Exact provider identity and ports staged; not approval.', provider=qualified,
                    scope_sha256=sha(reviews/'competition_otto_decomposed_complete_scope.json')))))

        ensure('artifact_versions', dict(version_id=version), dict(version_id=version, artifact_id=identity,
            content_hash=digest, semver='0.1.0+decomposed.'+digest[:12], is_latest=False, trust_tier=3,
            derives_from=UUID(scope['original_version_id'])))
        for table, rows, keys in [('artifact_cdg_nodes',nodes,['node_id']),
                                  ('artifact_cdg_edges',edges,['source_id','target_id','output_name','input_name'])]:
            for item in rows:
                row = dict(version_id=version, **item)
                ensure(table, {k:row[k] for k in ['version_id',*keys]}, row)
        ports(identity, version, boundary)
        for node in graph.nodes:
            fqdn = by_runtime[node.matched_primitive]
            qualified = scope['providers'][fqdn]
            ensure('artifact_cdg_bindings', dict(version_id=version,node_id=node.node_id), dict(
                version_id=version, node_id=node.node_id, bound_artifact_fqdn=fqdn,
                bound_version_content_hash=qualified['content_hash'], binding_confidence=1., binding_source=RUNNER,
                status='active', alternatives=Jsonb([]), evidence_summary=Jsonb(dict(
                    runtime_fqdn=node.matched_primitive, provider_version_id=qualified['version_id'],
                    source_sha256=qualified['source_sha256'], output_aliases_by_ordinal=[p.name for p in node.outputs]))))
        key = dict(dependent_version_id=version, dependency_artifact_fqdn=source['fqdn'],
                   dependency_content_hash=scope['original_content_hash'], port_name='')
        ensure('artifact_dependencies', key, dict(**key, dependency_role='cdg', optional=False,
            binding_metadata=Jsonb(dict(source_version_id=scope['original_version_id'],
                scope='Immutable original intake provenance; numerical execution uses direct atom bindings.'))))
        evidence = uuid5(version, RUNNER)
        ensure('artifact_audit_evidence', dict(evidence_id=evidence), dict(evidence_id=evidence, artifact_id=identity,
            version_id=version, audit_type='asset_integrity_check', passed=True, status='completed', source_kind='automated',
            runner_version=RUNNER, details=Jsonb(dict(scope=scope, approved=False, stager_sha256=sha(__file__)))))
        document = db.execute('SELECT get_artifact_document(%s) AS d', (source['fqdn'],)).fetchone()['d']
        restored = _artifact_document_to_cdg(document, version_id=str(version), content_hash=digest,
                                            require_execution_envelope=True)
        require(restored==graph, 'Stored graph differs from qualified graph')
        require(history(db,scope['original_version_id'])==scope['original_history_sha256'], 'Original history changed')
        latest = db.execute('SELECT version_id::text FROM artifact_versions WHERE artifact_id=%s AND is_latest',
                            (identity,)).fetchall()
        require(latest==[dict(version_id=scope['original_version_id'])], 'Original latest selection changed')
        require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=ANY(%s::uuid[])',
                               ([identity,*provider_ids],)).fetchone(), 'Draft candidates unexpectedly served')
        require(review()==scope, 'Qualification changed during staging')
        if not apply:
            db.rollback()
    return dict(applied=apply, approved=False, rows_created=created, providers=27, nodes=len(nodes), edges=len(edges),
                artifact_id=str(identity), version_id=str(version), graph_sha256=digest,
                original_history_preserved=True, original_latest_preserved=True, catalog_roundtrip=True,
                stager_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    result=stage(args.apply)
    suffix = ('import' if result['rows_created'] else 'repeat') if args.apply else 'rollback'
    (ROOT/'docs/reviews'/('competition_otto_decomposed_draft_'+suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
