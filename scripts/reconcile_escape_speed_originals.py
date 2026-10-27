"""Read-only source reconciliation and fresh escape execution; no promotion."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.audit_legacy_physics_execution_scope import scope
from scripts.audit_physics_projection_lineage import signature, digest
from scripts.audit_physics_legacy_reconciliation import source_inventory
from scripts.stage_remaining_legacy_physics_revisions import ROOT, sha, require, history
from scripts.validate_escape_speed_immutable_source import validate as validate_source
from scripts import validate_escape_speed_execution as oracle
from scripts.verify_escape_speed_catalog import verify


async def numerical(source):
    with patch.object(oracle, 'validate_source', side_effect=validate_source):
        result = await oracle.validate(ROOT, source/'symbols.cypher', source/'infrules.cypher', source/'expr_and_feed.cypher')
    normalized = json.loads(json.dumps(result))
    normalized['source_proof'].pop('immutable_validator_sha256')
    require(normalized == json.loads((ROOT/'docs/reviews/escape_speed_execution.json').read_text()),
            'Fresh escape execution differs from retained evidence')
    return result


def reconcile(source):
    execution = asyncio.run(numerical(source))
    parent = verify(ROOT)
    pairs = [g for g in json.loads((ROOT/'docs/reviews/physics_remaining_source_pairs.json').read_text())['graphs']
             if g['derivation_ids'] == ['129143']]
    require(len(pairs) == 1, 'Unique escape source pair required')
    pair = pairs[0]
    forms, inventories, originals = [], [], []
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for key in ('legacy', 'projected'):
            version = pair[key+'_version_id']
            row = db.execute('SELECT artifact_id::text,version_id::text,content_hash FROM artifact_versions WHERE version_id=%s', (version,)).fetchone()
            require(row == dict(artifact_id=pair[key+'_artifact_id'], version_id=version, content_hash=pair[key+'_content_hash']),
                    'Original escape identity changed')
            nodes = db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s', (version,)).fetchall()
            edges = db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s', (version,)).fetchall()
            bindings = db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s', (version,)).fetchall()
            forms.append(scope(nodes, edges))
            inventories.append(source_inventory([json.loads(n['type_signature']) for n in nodes], bindings))
            originals.append(dict(kind=key, **row, nodes=len(nodes), steps=len(forms[-1][0]), bindings=len(bindings),
                                  original_history_sha256=history(db, version)))
    a, b = forms
    require(signature(a[0]) == signature(b[0]), 'Grouped step contracts differ')
    require({k: Counter(v) for k,v in a[1].items()} == {k: Counter(v) for k,v in b[1].items()}, 'Pair multiplicities differ')
    require(a[2] == b[2], 'Dependency edges differ')
    x, y = inventories
    require(x['relations'] == y['relations'] and x['binding_versions'] == y['binding_versions']
            and x['all_bindings_active'] and y['all_bindings_active'], 'Inference or binding contracts differ')
    return dict(passed=True, read_only=True, approved=False, eligible_for_approval=False, catalog_mutations=0,
                originals=originals, shared_source_steps=len(a[0]), complete_pair_multiplicity=True,
                complete_dependency_edges=True, shared_scope_sha256=digest(signature(a[0])),
                fresh_execution=execution, existing_parent_verification=parent,
                limitations=['Source equality is not proof validity or original-identity promotion.',
                    'Reuse preserves the exact approved infall provider with explicit positional output aliases.',
                    'Positive escape threshold and launch energy; negative velocity is only the inward counterpart.',
                    'Fixed exterior spherical Newtonian field, negligible test mass and asymptotic rest at infinity; caller establishes collision-free outward trajectory.',
                    'Before promotion: exact parent/provider port and bound checks, corrected versions, stored execution, corruption rejection, rollback and fresh served execution.'],
                reconciler_sha256=sha(__file__),
                immutable_validator_sha256=sha(ROOT/'scripts/validate_escape_speed_immutable_source.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    report = reconcile(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/escape_speed_original_reconciliation.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    print(json.dumps(dict(passed=True, approved=False, shared_steps=report['shared_source_steps'], catalog_mutations=0)))
