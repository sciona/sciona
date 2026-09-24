"""Exercise approval rollback including failure after the final status mutation."""
import hashlib
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import scripts.promote_available_feature_providers as publisher
from scripts.plan_available_feature_providers import ROOT, plan, sha
from scripts.review_available_feature_publication import require
from scripts.validate_available_feature_staging import snapshot as staging_snapshot


def snapshot(proposed):
    result = staging_snapshot(proposed)
    ids = [atom['artifact_id'] for atom in proposed['atoms']] + [proposed['source_provenance']['artifact_id']]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for table, key in [('artifact_references', 'artifact_id'), ('atom_references', 'atom_id'),
                           ('artifact_audit_rollups', 'artifact_id'), ('atom_audit_rollups', 'atom_id')]:
            rows = db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s::uuid[]) ORDER BY to_jsonb(t)::text', (ids,)).fetchall()
            result[table] = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
        result['reference'] = db.execute('SELECT to_jsonb(t) AS row FROM references_registry t WHERE ref_id=%s', (publisher.REFERENCE,)).fetchall()
    return result


def validate():
    proposed = plan()
    before = snapshot(proposed)
    dry = publisher.promote()
    require(not dry['already_approved'] and dry['rows_created'] == 66 and dry['mutations'] == 92,
            'Expected fresh publication writes differ')
    require(snapshot(proposed) == before, 'Dry rollback changed catalog')
    checked = []
    for step in [1, 33, 66, 67, 92]:
        try:
            publisher.promote(fault_after=step)
        except RuntimeError as error:
            require(str(error) == 'Injected provider publication failure', 'Unexpected failure')
        else:
            raise ValueError('Publication fault did not fire')
        require(snapshot(proposed) == before, 'Publication fault changed catalog')
        checked.append(step)
    return dict(passed=True, rollback_verified=True, committed_catalog_mutations=0,
        dry_run=dry, failure_after_mutation_checks=checked,
        publisher_sha256=sha(Path(publisher.__file__)), validator_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/available_feature_provider_publication_gates.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, mutations=report['dry_run']['mutations'],
                         failure_checks=report['failure_after_mutation_checks'], committed_catalog_mutations=0)))
