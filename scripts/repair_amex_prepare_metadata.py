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
from scripts.review_amex_execution import audit, ROOT, sha


def require(condition, message):
    if not condition:raise ValueError(message)


def review():
    current=audit()
    require(current==json.loads((ROOT/'docs/reviews/competition_amex_semantic_review.json').read_text()), 'Original qualified semantic evidence changed')
    return current
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

IDENTITY = UUID('798a9f43-4f51-54c2-8ee2-32028cc9a852')
RUNNER = 'amex-prepare-metadata-repair.v1'
DESCRIPTION = ('Check caller supplied sequences, labels and settings before training. Require matching input sizes, '
    'valid category maps and enough examples for every training fold. Keep a canonical copy of the input '
    'and a fingerprint so later stages can detect changes. No models are trained here.')
CONTRACT = ('Finite version-1 JSON with exactly training, query and controls alongside version. Each population '
    'supplies sequences and unique opaque identities; training also supplies binary integer labels. Identities '
    'must be disjoint between populations. Raw sequences use runtime numeric, categorical, time and month roles, '
    'one through thirteen rows per sequence and matching numeric widths. Null represents missing numeric data. '
    'Category rules are explicit numeric or string-to-integer maps with bounded codes and explicit missing policy. '
    'Time values must be finite; month grouping values are integers. Controls specify category_rules, distinct valid '
    'zero_fill_columns and an integer seed in [0,2**31). Each class needs at least five examples and every shuffled '
    'stratified five-fold training split must hold at least 256 examples. Canonical finite JSON is limited to '
    '64 MiB. Preprocessing validation rejects invalid roles, categories, widths and arithmetic overflow. Return '
    'an immutable Prepared object holding canonical JSON and its integrity fingerprint. This stage validates '
    'and encodes input; it does not train models or establish source data authenticity. Disjoint identities do '
    'not detect duplicate content with changed identities.')


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
    record=dict(version_id='ee9c0539-0043-54e3-a21f-bada18a14a04', content_hash='c81448564116ed6ef2b4162829f50d9c06efb3479dc02c5e26a826b0e6127fc6', provider_sha256=reviewed['provider_sha256'])
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
                prior_source_review_sha256=sha(ROOT/'docs/reviews/competition_amex_semantic_review.json'),
                repair_script_sha256=sha(ROOT/'scripts/repair_amex_prepare_metadata.py')))))
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
                script_sha256=sha(ROOT/'scripts/repair_amex_prepare_metadata.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    print(json.dumps(repair(apply=parser.parse_args().apply), indent=2))
