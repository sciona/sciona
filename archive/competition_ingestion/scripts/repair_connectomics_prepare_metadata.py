"""Complete missing metadata for the already approved, unchanged prepare provider.

Default transaction rolls back. Conflicting rows fail closed. No approval,
version, binding, reference or prior audit is changed.
"""
import argparse
import json
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import ensure_row, comparable
from scripts.review_decomposed_connectomics import review, ROOT, sha, require
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

IDENTITY = UUID('8e88bf04-bd23-5a25-9e9c-f00ee6f3665a')
RUNNER = 'connectomics-prepare-metadata-repair.v1'
DESCRIPTION = ('Check and copy the input table without changing the caller’s values. '
    'The caller chooses a method and whether to add the time order score. '
    'Values must be finite and nonnegative, with at least three columns and more rows than columns. '
    'Reject constant columns and values that cannot be stored as finite 32 bit numbers.')
CONTRACT = ('A dictionary with exactly version, signals, mode and directivity. Version is integer 1 '
    '(not bool); mode is simple or tuned; directivity is a bool. Signals are a real numeric 2D '
    'time-by-node matrix with nodes >= 3 and samples >= max(5, nodes+1). Conversion to float32 '
    'must remain finite and nonnegative; every channel must vary. Returns a read-only Fortran '
    'float32 copy in Prepared. No inference, normalization or resampling occurs here.')


class InjectedFailure(RuntimeError):
    pass


def protected(db):
    result = {}
    for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
        for suffix in ('s', '_versions', '_io_specs', '_references', '_audit_rollups'):
            table = prefix + suffix
            # Metadata triggers touch the entity timestamp; substantive fields
            # and every version/history record must remain unchanged.
            expression = "to_jsonb(t)-'updated_at'" if suffix == 's' else 'to_jsonb(t)'
            rows = db.execute(f'SELECT {expression} AS row FROM {table} t WHERE {key}=%s ORDER BY to_jsonb(t)::text', (IDENTITY,)).fetchall()
            result[table] = comparable(rows)
    rows = db.execute('''SELECT to_jsonb(t) AS row FROM artifact_audit_evidence t
        WHERE artifact_id=%s AND runner_version IS DISTINCT FROM %s ORDER BY to_jsonb(t)::text''', (IDENTITY, RUNNER)).fetchall()
    result['prior_evidence'] = comparable(rows)
    return result


def repair(apply=False, fail_after=None):
    reviewed = review()
    record, = [r for r in reviewed['providers'] if r['runtime_fqdn'].endswith('.connectomics_prepare')]
    version = UUID(record['version_id'])
    score = round(estimate_jargon_score(DESCRIPTION), 6)
    require(score < .4, 'Plain-language description exceeds jargon gate')
    writes = 0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        before = protected(db)
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            row = db.execute(f'''SELECT a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest
                FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                WHERE a.{key}=%s AND v.version_id=%s FOR UPDATE OF a,v''', (IDENTITY, version)).fetchone()
            require(row == dict(status='approved', is_publishable=True, content_hash=record['content_hash'],
                                trust_tier=3, is_latest=True), 'Existing approved identity/version differs')
            ports = db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr
                FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
            require(ports == [dict(direction='input', name='payload', ordinal=0, type_desc='dict', required=True, default_value_repr=''),
                              dict(direction='output', name='prepared', ordinal=0, type_desc='object', required=True, default_value_repr='')],
                    'Existing interface differs')

        def ensure(table, keys, row):
            nonlocal writes
            writes += int(ensure_row(db, table, keys, row))
            if fail_after == writes:
                raise InjectedFailure('Injected metadata repair rollback')

        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            for kind, content in [('technical', CONTRACT), ('dejargonized', DESCRIPTION)]:
                keys = {key: IDENTITY, 'kind': kind, 'language': 'en'}
                ensure(prefix+'_descriptions', keys, dict(keys, content=content, generated_by=RUNNER,
                    reviewed=False, jargon_score=round(estimate_jargon_score(content), 6)))
            keys = {key: IDENTITY, 'version_id': version, 'name': 'payload'}
            ensure(prefix+'_parameters', keys, dict(keys, type_desc='dict', required=True,
                default_value_repr='', position=0, kind='positional_or_keyword',
                technical_description=CONTRACT, dejargonized_description=DESCRIPTION,
                constraints_json=Jsonb(dict(validity_bounds=CONTRACT, review_status='automated_tier_3_pass_with_limits'))))
        evidence_id = uuid5(version, RUNNER)
        ensure('artifact_audit_evidence', {'evidence_id': evidence_id}, dict(evidence_id=evidence_id,
            artifact_id=IDENTITY, version_id=version, audit_type='semantic_audit', passed=True,
            status='completed', source_kind='automated', runner_version=RUNNER,
            details=Jsonb(dict(scope='Missing description and parameter metadata only; prior approval unchanged.',
                implementation_sha256=record['provider_sha256'], publication_tier=3,
                prior_source_review_sha256=sha(ROOT/'docs/reviews/competition_connectomics_semantic_review.json'),
                repair_script_sha256=sha(ROOT/'scripts/repair_connectomics_prepare_metadata.py')))))
        require(protected(db) == before, 'Prior provider history or approval changed')
        eligible = db.execute('SELECT atom_is_publishable(%s) AS atom,artifact_is_publishable(%s) AS artifact',
                              (IDENTITY, IDENTITY)).fetchone()
        require(eligible == dict(atom=True, artifact=True), 'Metadata repair did not restore native eligibility')
        require(review() == reviewed, 'Review evidence changed during repair')
        if fail_after is not None:
            raise ValueError('Requested failure injection was not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply, rows_created=writes, native_eligible=eligible,
                existing_approval_and_history_preserved=True, new_provider_approvals=0,
                script_sha256=sha(ROOT/'scripts/repair_connectomics_prepare_metadata.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    print(json.dumps(repair(apply=parser.parse_args().apply), indent=2))
