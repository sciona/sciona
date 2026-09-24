"""Atomically approve the qualified nasa-lifecycle drafts at automated Tier 3."""
import argparse
import json
from pathlib import Path
from uuid import UUID, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.cdg_projection import build_published_cdg_projection
from scripts.import_residual_execution_drafts import ensure_row
from scripts.plan_nasa_lifecycle_drafts import plan, BUILDERS
from scripts.review_conditional_correction import ROOT, require, sha
from scripts.review_nasa_lifecycle_publication import review, LIMITATIONS
from scripts.validate_nasa_lifecycle_database_gates import check_staged

RUNNER = 'nasa-lifecycle-community.v1'


def promote(apply=False):
    if apply:
        gate_path = ROOT / 'docs/reviews/nasa_lifecycle_publication_transaction_gates.json'
        gates = json.loads(gate_path.read_text())
        require(gates['passed'] is True and gates['publisher_sha256'] == sha(Path(__file__))
                and gates['validator_sha256'] == sha(ROOT / 'scripts/validate_nasa_lifecycle_publication_transaction.py')
                and gates['injected_transaction_failure_rolled_back'] is True, 'Publication transaction gates missing or stale')
    qualification, proposed = review(), plan()
    graph_targets = list(proposed['graphs'].values())
    topologies = {}
    for kind, graph in proposed['graphs'].items():
        topologies[kind] = build_published_cdg_projection(
            artifact={'artifact_id': graph['artifact_id'], 'fqdn': graph['fqdn']},
            version={'version_id': graph['version_id'], 'content_hash': graph['graph_sha256']},
            cdg=BUILDERS[kind]()).topo_hash
    created = 0
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('residual-classifier-draft.v1'))")
        db.execute("SELECT pg_advisory_xact_lock(hashtext('nasa-domain-draft.v1'))")
        db.execute("SELECT pg_advisory_xact_lock(hashtext('nasa-lifecycle-draft.v1'))")
        ids = [UUID(b['artifact_id']) for b in graph_targets + proposed['atoms']]
        states = db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=ANY(%s::uuid[]) FOR UPDATE', (ids,)).fetchall()
        require(len(states) == len(ids), 'Staged artifacts missing')
        already = all(r['status'] == 'approved' and r['is_publishable'] for r in states)
        check_staged(db, proposed, approved=already)

        def ensure(table, keys, row):
            nonlocal created
            created += ensure_row(db, table, keys, row)

        reference = 'nasa-lifecycle-drivendata-source'
        url = 'https://github.com/drivendataorg/nasa-airport-pushback/tree/09f2f5d2940dd6b63b93115a0e9fb9e8a964c700'
        title = 'Pinned NASA third-place nasa-lifecycle source'
        ensure('references_registry', {'ref_id': reference}, dict(ref_id=reference, ref_type='web', title=title, url=url))
        rollup = dict(overall_verdict='acceptable_with_limits', structural_status='pass', runtime_status='pass',
            semantic_status='pass', developer_semantics_status='pass', review_status='approved', review_semantic_verdict='pass',
            review_developer_semantics_verdict='pass', trust_readiness='ready', review_limitations=LIMITATIONS,
            review_required_actions=[], trust_blockers=[], acceptability_band='acceptable_with_limits',
            parity_coverage_level='positive_and_negative', parity_test_status='pass')
        targets = [(UUID(b['artifact_id']), UUID(b['version_id']), b) for b in proposed['atoms']] + [(UUID(g['artifact_id']), UUID(g['version_id']), None) for g in graph_targets]
        for target, selected, binding in targets:
            for table, key in [('artifact_references', 'artifact_id')] + ([('atom_references', 'atom_id')] if binding else []):
                ensure(table, {key: target, 'ref_key': reference}, {key: target, 'ref_key': reference, 'ref_id': reference,
                    'title': title, 'url': url, 'source': 'llm_extracted', 'verified': True, 'confidence': 'high',
                    'relevance_note': 'Pinned source provenance; separate training and inference, complete private runtime state, retained MIT notice and synthetic source parity.'})
            for table, key in [('artifact_audit_rollups', 'artifact_id')] + ([('atom_audit_rollups', 'atom_id')] if binding else []):
                ensure(table, {key: target}, {key: target, **rollup})
            evidence = uuid5(selected, RUNNER)
            details = dict(qualification=qualification, publisher_sha256=sha(Path(__file__)), publication_tier=3,
                           scope='Provisioned in-process paired training and inference with complete private runtime state; shared numerical atoms remain reusable across domains', binding=binding)
            # Omit changing catalog-state observations from immutable evidence.
            if binding:
                details['binding'] = {k: v for k, v in binding.items() if not k.startswith('existing_')}
            ensure('artifact_audit_evidence', {'evidence_id': evidence}, dict(evidence_id=evidence, artifact_id=target,
                version_id=selected, audit_type='semantic_audit', passed=True, status='completed', source_kind='automated',
                runner_version=RUNNER, details=Jsonb(details)))
        for binding in proposed['atoms']:
            target = UUID(binding['artifact_id'])
            db.execute("UPDATE atoms SET status='approved',is_publishable=true WHERE atom_id=%s AND status<>'approved'", (target,))
            db.execute("UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s AND status<>'approved'", (target,))
            require(db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s', (target,)).fetchone(), 'Provider not served')
        if not already:
            for kind, graph in proposed['graphs'].items():
                db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=%s,"
                    'top_level_input_arity=%s,top_level_output_arity=%s,topo_hash=%s WHERE artifact_id=%s',
                    (len(BUILDERS[kind]().nodes), len(graph['boundary_inputs']), len(graph['boundary_outputs']),
                     topologies[kind], graph['artifact_id']))
                db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s',
                    (graph['version_id'], graph['artifact_id']))
        check_staged(db, proposed, approved=True)
        count = db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served '
            'WHERE artifact_id=ANY(%s::uuid[])', (ids,)).fetchone()['n']
        require(count == len(ids), 'New atoms and paired graphs must all be served')
        require(review() == qualification, 'Qualification changed during transaction')
        if not apply:
            db.rollback()
    return dict(applied=apply, already_approved=already, rows_created=created, trust_tier=3,
                served_atoms_in_transaction=len(proposed['atoms']), served_cdgs_in_transaction=len(graph_targets),
                version_ids={k: g['version_id'] for k, g in proposed['graphs'].items()},
                graph_sha256={k: g['graph_sha256'] for k, g in proposed['graphs'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = promote(args.apply)
    if args.apply:
        report = 'nasa_lifecycle_publication_repeat.json' if result['already_approved'] else 'nasa_lifecycle_publication.json'
        (ROOT / 'docs/reviews' / report).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
