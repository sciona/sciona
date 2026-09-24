"""Atomically publish the qualified reusable materialized providers at Tier 3."""
import argparse
import json
from pathlib import Path
from uuid import UUID, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from scripts.import_residual_execution_drafts import ensure_row
from scripts.review_nasa_first_lifecycle_publication import ROOT, plan, sha
from scripts.review_nasa_first_lifecycle_publication import require, review
from scripts.stage_nasa_first_lifecycle_providers import RUNNER as STAGING_RUNNER
from scripts.validate_nasa_first_lifecycle_database_gates import check_staged

RUNNER = 'nasa-first-lifecycle-provider-community.v1'
REFERENCE = 'nasa-first-lifecycle-pinned-source'


def promote(apply=False, fault_after=None):
    if apply:
        gate = json.loads((ROOT/'docs/reviews/competition_nasa_first_lifecycle_publication_gates.json').read_text())
        require(gate['passed'] and gate['publisher_sha256'] == sha(Path(__file__))
                and gate['validator_sha256'] == sha(ROOT/'scripts/validate_nasa_first_lifecycle_publication.py')
                and gate['rollback_verified'], 'Publication gates missing or stale')
    qualification, proposed = review(), plan()
    created = 0
    mutations = 0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (STAGING_RUNNER,))
        ids = [UUID(atom['artifact_id']) for atom in proposed['atoms']]
        states = db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=ANY(%s::uuid[]) FOR UPDATE', (ids,)).fetchall()
        require(len(states) == 35, 'Staged artifacts missing')
        already = all(row['status'] == 'approved' and row['is_publishable'] for row in states)
        check_staged(db, proposed, approved=already)

        def written(count):
            nonlocal mutations
            mutations += count
            if fault_after is not None and mutations >= fault_after:
                raise RuntimeError('Injected provider publication failure')

        def ensure(table, keys, row):
            nonlocal created
            count = ensure_row(db, table, keys, row)
            created += count
            written(int(count))

        url = 'https://github.com/drivendataorg/nasa-airport-pushback/tree/09f2f5d2940dd6b63b93115a0e9fb9e8a964c700'
        title = 'Pinned first-place source motivating reusable lifecycle operations and explicit domain adapters'
        ensure('references_registry', {'ref_id': REFERENCE}, dict(ref_id=REFERENCE, ref_type='web', title=title, url=url))
        rollup = dict(overall_verdict='acceptable_with_limits', structural_status='pass', runtime_status='pass',
            semantic_status='pass', developer_semantics_status='pass', review_status='approved', review_semantic_verdict='pass',
            review_developer_semantics_verdict='pass', trust_readiness='ready', review_limitations=qualification['limitations'],
            review_required_actions=[], trust_blockers=[], acceptability_band='acceptable_with_limits',
            parity_coverage_level='positive_and_negative', parity_test_status='pass')
        for atom in proposed['atoms']:
            identity, version = UUID(atom['artifact_id']), UUID(atom['version_id'])
            for table, key in [('artifact_references', 'artifact_id'), ('atom_references', 'atom_id')]:
                ensure(table, {key: identity, 'ref_key': REFERENCE}, {key: identity, 'ref_key': REFERENCE,
                    'ref_id': REFERENCE, 'title': title, 'url': url, 'source': 'llm_extracted', 'verified': True,
                    'confidence': 'high', 'relevance_note': 'Corrected lifecycle provider with explicit role and port contracts; generic operation transfer evidence is distinct from source-specific adapter qualification.'})
            for table, key in [('artifact_audit_rollups', 'artifact_id'), ('atom_audit_rollups', 'atom_id')]:
                ensure(table, {key: identity}, {key: identity, **rollup})
            evidence = uuid5(version, RUNNER)
            ensure('artifact_audit_evidence', {'evidence_id': evidence}, dict(evidence_id=evidence,
                artifact_id=identity, version_id=version, audit_type='semantic_audit', passed=True, status='completed',
                source_kind='automated', runner_version=RUNNER,
                details=Jsonb(dict(qualification=qualification, binding=atom, publication_tier=3,
                                  publisher_sha256=sha(Path(__file__))))))
        for atom in proposed['atoms']:
            for table, key in [('artifacts', 'artifact_id'), ('atoms', 'atom_id')]:
                written(db.execute(f"UPDATE {table} SET status='approved',is_publishable=true WHERE {key}=%s AND (status<>'approved' OR NOT is_publishable)",
                                   (atom['artifact_id'],)).rowcount)
        check_staged(db, proposed, approved=True)
        for view, key in [('catalog_artifacts_served', 'artifact_id'), ('catalog_atoms_served', 'atom_id')]:
            require(db.execute(f'SELECT count(DISTINCT {key}) AS n FROM {view} WHERE {key}=ANY(%s::uuid[])', (ids,)).fetchone()['n'] == 35,
                    'Published provider serving incomplete')
        require(review() == qualification, 'Qualification changed during publication')
        if not apply:
            db.rollback()
    return dict(applied=apply, already_approved=already, rows_created=created, mutations=mutations,
                trust_tier=3, served_atoms_in_transaction=35, original_workflow_approved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(promote(args.apply)))
