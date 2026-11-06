"""Check rollback and optionally apply the narrow approved-provider repair."""
import argparse
import hashlib
import json
from scripts.repair_decomposed_aptos_metadata import (
    IDENTITIES, InjectedFailure, repair, protected, connect, ROOT, sha)


def snapshot():
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        state = protected(db)
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            for suffix in ('descriptions', 'parameters', 'audit_rollups'):
                table = prefix+'_'+suffix
                state[table] = db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text', (IDENTITIES,)).fetchall()
        state['all_evidence'] = db.execute('SELECT to_jsonb(t) AS row FROM artifact_audit_evidence t WHERE artifact_id=ANY(%s) ORDER BY to_jsonb(t)::text', (IDENTITIES,)).fetchall()
        return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    apply = parser.parse_args().apply
    before = snapshot()
    dry_run = repair()
    assert snapshot() == before
    failures = []
    if dry_run['rows_changed']:
        assert dry_run['rows_changed'] == 46
        for step in (1, 10, 23, 46):
            try:
                repair(fail_after=step)
            except InjectedFailure:
                failures.append(step)
            else:
                raise AssertionError('Failure injection was not reached')
            assert snapshot() == before
    report = dict(passed=True, applied=False, dry_run=dry_run, rollback_steps=failures,
                  dry_run_preserved_state=True, no_new_approvals=True,
                  validator_sha256=sha(ROOT/'scripts/validate_decomposed_aptos_metadata_repair.py'))
    if apply:
        result = repair(apply=True)
        repeat = repair(apply=True)
        assert repeat['rows_changed'] == 0
        report.update(applied=True, result=result, repeat=repeat)
    name = 'decomposed_aptos_metadata_repair.json' if apply else 'decomposed_aptos_metadata_repair_gates.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
