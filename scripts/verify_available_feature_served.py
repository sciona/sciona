"""Fresh read-only verification of approved providers and immutable review evidence."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.plan_available_feature_providers import ROOT, plan, sha
from scripts.promote_available_feature_providers import RUNNER, REFERENCE
from scripts.review_available_feature_publication import require, review
from scripts.validate_available_feature_database_gates import check_staged


def verify():
    proposed, qualification = plan(), review()
    rows = []
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db, proposed, approved=True)
        for atom in proposed['atoms']:
            for view, key in [('catalog_atoms_served', 'atom_id'), ('catalog_artifacts_served', 'artifact_id')]:
                served = db.execute(f'SELECT fqdn,review_status,trust_readiness,reference_count FROM {view} WHERE {key}=%s',
                                    (atom['artifact_id'],)).fetchall()
                require(served and all(row == dict(fqdn=atom['fqdn'], review_status='approved',
                    trust_readiness='ready', reference_count=1) for row in served), 'Served review metadata differs')
            audits = db.execute('SELECT details,status,passed,source_kind FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s',
                                (atom['version_id'], RUNNER)).fetchall()
            expected = dict(details=dict(qualification=qualification, binding=atom, publication_tier=3,
                publisher_sha256=sha(ROOT/'scripts/promote_available_feature_providers.py')),
                status='completed', passed=True, source_kind='automated')
            require(audits == [expected], 'Published immutable qualification differs')
            for table, key in [('artifact_references', 'artifact_id'), ('atom_references', 'atom_id')]:
                refs = db.execute(f'SELECT ref_id,verified FROM {table} WHERE {key}=%s', (atom['artifact_id'],)).fetchall()
                require(refs == [dict(ref_id=REFERENCE, verified=True)], 'Published source reference differs')
            rows.append({key: atom[key] for key in ['artifact_id', 'version_id', 'content_hash', 'runtime_fqdn']})
        source = db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',
                            (proposed['source_provenance']['artifact_id'],)).fetchone()
        require(source == dict(status='draft', is_publishable=False), 'Original workflow publication changed')
    return dict(passed=True, read_only=True, approved=True, trust_tier=3, served_providers=len(rows),
        providers=rows, original_workflow_approved=False, stored_ports_and_runtime_bindings_verified=True,
        qualification=qualification, verifier_sha256=sha(Path(__file__)),
        limitations=['Fresh SQL served-view, version, contract and approval verification; not HTTP transport or ranking qualification.',
            'Individual provider graph execution is bound in the review; full original workflow graph execution remains pending.'])


if __name__ == '__main__':
    report = verify()
    (ROOT/'docs/reviews/available_feature_provider_served.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, served_providers=report['served_providers'], trust_tier=3, original_workflow_approved=False)))
