"""Reconcile one retained rule-arity correction without rewriting either history."""
import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.audit_legacy_physics_execution_scope import scope
from scripts.audit_physics_projection_lineage import signature, digest
from scripts.audit_physics_legacy_reconciliation import source_inventory
from scripts.stage_remaining_legacy_physics_revisions import ROOT, sha, require, history
from scripts.validate_orbit_radius_immutable_source import validate as validate_source
from scripts import validate_orbit_radius_execution as oracle
from scripts.verify_orbit_radius_catalog import verify
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts

STEP = ('282755', '1306821')
LEGACY_RULE = '111777'
PROJECTED_RULE = '111236'
FEEDS = ['5089196493', '1333474099', '3342155559', '2114570475', '7912578203', '9789485295']


async def numerical(source):
    with patch.object(oracle, 'validate_source', side_effect=validate_source):
        result = await oracle.validate(ROOT, source/'symbols.cypher', source/'infrules.cypher', source/'expr_and_feed.cypher')
    normalized = json.loads(json.dumps(result))
    normalized['source_proof'].pop('immutable_validator_sha256')
    require(normalized == json.loads((ROOT/'docs/reviews/orbit_radius_execution.json').read_text()),
            'Fresh orbit execution differs from retained evidence')
    return result


def public_step(source, execution):
    pins = execution['source_proof']['source_file_sha256']
    require(sha(source/'steps.cypher') == pins['conversion_of_data_formats/steps.cypher'], 'Public step file changed')
    rules = load_pinned_rule_contracts((source/'infrules.cypher').read_bytes(), pins['conversion_of_data_formats/infrules.cypher'])
    require(asdict(rules[LEGACY_RULE]) == dict(name='change four variables in expr', inputs=1, feeds=8, outputs=1), 'Legacy rule contract changed')
    require(asdict(rules[PROJECTED_RULE]) == dict(name='change three variables in expr', inputs=1, feeds=6, outputs=1), 'Projected rule contract changed')
    links = {}
    for block in re.split(r'(?m)^UNWIND ', (source/'steps.cypher').read_text()):
        if not block.startswith('[{start: {id:"'+STEP[1]+'"}'): continue
        target = re.search(r'end: \{id:"(\d+)"\}', block)
        relation = re.search(r'CREATE \(start\)-\[r:(\w+)\]', block)
        require(target is not None and relation is not None, 'Malformed public step relation')
        sequence = re.search(r'sequence_index:"?(\d+)"?', block)
        links.setdefault(relation[1], []).append((int(sequence[1]) if sequence else -1, target[1]))
    expected = dict(HAS_INFERENCE_RULE=[(-1, LEGACY_RULE)], HAS_INPUT=[(0, '9226945488')],
                    HAS_OUTPUT=[(0, '4627284246')], HAS_FEED=list(enumerate(FEEDS)))
    require({k: sorted(v) for k,v in links.items()} == expected, 'Public source step contract changed')
    return dict(step_id=STEP[1], legacy_rule=dict(id=LEGACY_RULE, **asdict(rules[LEGACY_RULE])),
                projected_rule=dict(id=PROJECTED_RULE, **asdict(rules[PROJECTED_RULE])),
                supplied_feed_count=6, feed_ids=FEEDS, source_step_file_sha256=sha(source/'steps.cypher'),
                original_arity_valid=False, projected_arity_valid=True)


def reconcile(source):
    execution = asyncio.run(numerical(source))
    correction = public_step(source, execution)
    parent = verify(ROOT)
    pairs = [g for g in json.loads((ROOT/'docs/reviews/physics_remaining_source_pairs.json').read_text())['graphs']
             if g['derivation_ids'] == ['282755']]
    require(len(pairs) == 1, 'Unique orbital-radius source pair required')
    pair = pairs[0]
    forms, inventories, originals = [], [], []
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for kind, rule in [('legacy', LEGACY_RULE), ('projected', PROJECTED_RULE)]:
            version = pair[kind+'_version_id']
            row = db.execute('SELECT artifact_id::text,version_id::text,content_hash FROM artifact_versions WHERE version_id=%s', (version,)).fetchone()
            require(row == dict(artifact_id=pair[kind+'_artifact_id'], version_id=version, content_hash=pair[kind+'_content_hash']), 'Original identity changed')
            before = history(db, version)
            nodes = db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s', (version,)).fetchall()
            edges = db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s', (version,)).fetchall()
            bindings = db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s', (version,)).fetchall()
            form = scope(nodes, edges)
            require(len(form[0]) == 13 and len(form[2]) == 12, 'Source cardinality changed')
            step = form[0][STEP]
            require(step['rule'] == rule and [f['feed_id'] for f in step['bindings']['feeds']] == FEEDS
                    and step['inputs'] == {'4b5c628c-6ada-543a-8fe0-f1dad3f66820'}
                    and step['outputs'] == {'10e3b342-67ce-5cff-9703-79e6977965df'}, 'Reviewed rule correction changed')
            signatures = [json.loads(n['type_signature']) for n in nodes]
            original_inventory = source_inventory(signatures, bindings)
            # Comparison normalization only: retain both exact historical forms.
            for sig in signatures:
                v = sig['variable_bindings']
                if (v['derivation_id'], v['step_id']) == STEP:
                    require(sig['inference_rule_id'] == rule, 'Unexpected split rule')
                    sig['inference_rule_id'] = PROJECTED_RULE
            normalized_inventory = source_inventory(signatures, bindings)
            forms.append(form); inventories.append((original_inventory, normalized_inventory))
            require(history(db, version) == before, 'Source history changed during reconciliation')
            originals.append(dict(kind=kind, **row, nodes=len(nodes), steps=len(form[0]), bindings=len(bindings),
                                  original_history_sha256=before, grouped_scope_sha256=digest(signature(form[0]))))
    a,b = forms
    sa,sb = signature(a[0]),signature(b[0])
    require({k for k in sa.keys() | sb.keys() if sa.get(k) != sb.get(k)} == {str(STEP)}, 'Unexpected grouped scope difference')
    sa[str(STEP)]['rule'] = PROJECTED_RULE
    require(sa == sb, 'Differences beyond the one reviewed rule ID')
    require({k:Counter(v) for k,v in a[1].items()} == {k:Counter(v) for k,v in b[1].items()} and a[2] == b[2], 'Pair multiplicities or dependency edges differ')
    x,y = (i[0] for i in inventories)
    require(x['relations'].keys() == y['relations'].keys(), 'Source relation inventory differs')
    changed = sorted(k for k in x['relations'] if x['relations'][k] != y['relations'][k])
    require(len(changed) == 1 and ':1306821:' in changed[0], 'Unexpected changed source contract')
    require(inventories[0][1]['relations'] == inventories[1][1]['relations'] and x['binding_versions'] == y['binding_versions']
            and x['all_bindings_active'] and y['all_bindings_active'], 'Normalized inference or binding contracts differ')
    return dict(passed=True, read_only=True, approved=False, eligible_for_approval=False, catalog_mutations=0,
                originals=originals, shared_source_steps=13, complete_pair_multiplicity=True, complete_dependency_edges=True,
                exact_original_rule_parity=False, explicit_rule_correction=correction, changed_contract_relations=changed,
                reconciled_scope_sha256=digest(sb), fresh_execution=execution, existing_parent_verification=parent,
                limitations=['Only the reviewed rule ID is normalized in memory for comparison; both histories remain unchanged.',
                    'The source four-variable rule has only three replacement pairs; the projected three-variable rule corrects its arity, without certifying literal source proof validity.',
                    'Newtonian circular exterior test-mass model; radius measured from center. Caller establishes clearance and geometry; period alone does not certify geostationarity.',
                    'Exact parent/provider qualification, corrected original versions, stored execution, corruption/rollback, publication and served checks remain required.'],
                reconciler_sha256=sha(__file__), immutable_validator_sha256=sha(ROOT/'scripts/validate_orbit_radius_immutable_source.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    report = reconcile(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/orbit_radius_original_reconciliation.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    print(json.dumps(dict(passed=True, approved=False, shared_steps=report['shared_source_steps'], catalog_mutations=0)))
