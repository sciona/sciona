"""Restore missing metadata for the existing approved M5 preparation atom."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5

from psycopg.types.json import Jsonb
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.review_decomposed_m5_implementation import review
from scripts.promote_nab_decomposed_providers import TABLES, snapshot
from scripts.review_decomposed_m5_implementation import ROOT, sha, require
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

RUNNER = 'm5-preparation-metadata-repair.v1'
IDENTITY = UUID('a24479e8-9672-5024-802d-d294c36f45d6')
GATES = ROOT/'docs/reviews/m5_preparation_metadata_gates.json'
PLAIN = 'Check private series history, group roles, dates, known covariates and training start days. Require unique series identities and a complete future grid. Return a private encoded copy. This step validates features but does not train models or measure forecast quality.'
CONTRACT = 'Validate an exact version-1 finite JSON payload with version, identities, configuration and controls. Require Python integer version one excluding bool. Configuration contains exactly history, first_day, pairs, roles, calendar, prices and groupings. Controls contain recursive_first_day and nonrecursive_first_day. Serialize with sorted keys, compact separators and allow_nan=False; reject encoded UTF-8 above 256 MiB before preprocessing. After canonical decoding require history to be a nonempty rectangular list of nonempty lists of finite nonnegative Python int/float values excluding bool. Require first_day, both start controls, pair codes, hierarchy role codes, grouping indices and calendar/price day/week/key codes to be signed 32-bit Python integers excluding bool. Identities are aligned unique nonempty strings; repeated contents with different identities are not detected. Calendar contains day, week, date and categories; prices contain pair, outlet, product, week and value. Calendar dates are ISO strings. Calendar categories contain scalar strings/numbers or null; each of the seven aligned category roles must be homogeneous strings or numbers apart from missing values. Price values are finite nonnegative numbers or null. Full preprocessing requires five aligned hierarchy roles, unique pair codes, eleven groupings in reviewed role order, unique calendar days covering all history and 28 future days, consistent one-to-one pair/outlet/product mappings, unique right join keys, and calendar week coverage for every price row. Each series must retain 28 future rows after release filtering. Price role mapping must agree with hierarchy roles for input pairs. Both training starts must be no later than the history cutoff. Build and validate complete feature blocks, including source range-based precision and finite float16 lag/encoding bounds; discard these temporary blocks at this boundary. Known future prices can participate in features and eligible training labels in target moments. This is not independent validation. Return frozen Prepared with canonical JSON and its SHA-256 integrity fingerprint. No model is trained, no forecast accuracy is established, and no original-CDG approval or historical software equivalence is implied. The fingerprint checks integrity, not authenticity; downstream intermediates must be from the same invocation. Pool-specific positive-target and training/diagnostic coverage checks occur during fitting.'


class InjectedFailure(RuntimeError):
    pass


def protected(db):
    # Protect the target's complete prior history and all other related records.
    plan = json.loads((ROOT/'docs/reviews/m5_decomposed_provider_intake_plan.json').read_text())
    ids = [UUID(r['atom_id']) for r in plan['providers']] + [UUID(x) for x in (
        '9c0a711a-dc75-5c7f-8711-b2879024e282', 'a059cde0-72a3-5866-8d2a-dec4f72587a6',
        '4dac38fe-1a5e-50ad-a112-47f1652a5dd0', '4f486f1c-82db-50db-b5d2-188aeaa8cc4a')]
    state = {}
    for table, key in TABLES:
        clause, args = f'{key}=ANY(%s)', [ids + [IDENTITY]]
        expression = 'to_jsonb(t)'
        if table in ('atoms', 'artifacts'):
            # Description insertion touches the target parent timestamp through
            # a database trigger. Preserve every other field and other parent.
            expression = f"CASE WHEN {key}='{IDENTITY}'::uuid THEN to_jsonb(t)-'updated_at' ELSE to_jsonb(t) END"
        if table.endswith(('_descriptions', '_parameters')):
            args = [ids]
        if table == 'artifact_audit_evidence':
            clause += ' AND NOT (artifact_id=%s AND runner_version=%s)'
            args += [IDENTITY, RUNNER]
        state[table] = db.execute(f'SELECT {expression} AS row FROM {table} t WHERE {clause} ORDER BY to_jsonb(t)::text', args).fetchall()
    for table in ('artifact_cdg_nodes', 'artifact_cdg_edges', 'artifact_cdg_bindings', 'artifact_dependencies'):
        key = 'dependent_version_id' if table == 'artifact_dependencies' else 'version_id'
        state[table] = db.execute(f'''SELECT to_jsonb(t) AS row FROM {table} t WHERE {key} IN
            (SELECT version_id FROM artifact_versions WHERE artifact_id=ANY(%s)) ORDER BY to_jsonb(t)::text''', (ids + [IDENTITY],)).fetchall()
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


def repair(apply=False, fail_after=None):
    qualification = review()
    path = ROOT/'docs/reviews/decomposed_m5_implementation_review.json'
    integration = json.loads(path.read_text())
    require(qualification == integration, 'Integration review differs')
    records = [r for r in integration['providers'] if r['runtime_fqdn'].endswith('.m5_prepare')]
    require(len(records) == 1, 'Unique qualified preparation provider required')
    record = records[0]
    version = UUID(record['version_id'])
    if apply:
        gates = json.loads(GATES.read_text())
        require(gates['passed'] and gates['all_rollbacks_complete'] and
                gates['script_sha256'] == sha(Path(__file__)) and gates['review_sha256'] == sha(path),
                'Matching metadata repair gates required')
    created = 0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        before = protected(db)
        def ensure(table, keys, row):
            nonlocal created
            created += int(ensure_row(db, table, keys, row))
            if fail_after is not None and created == fail_after:
                raise InjectedFailure('Injected metadata repair failure')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (version,)).fetchone(), 'Failed provider audit exists')
        require(not db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (version,)).fetchone(), 'Unexpected internal provider dependency')
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            actual = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest FOR UPDATE OF a,v''', (IDENTITY,)).fetchall()
            require(actual == [dict(fqdn=record['catalog_fqdn'], status='approved', is_publishable=True,
                version_id=str(version), content_hash=record['content_hash'], trust_tier=3)], 'Exact approved preparation version differs')
            ports = db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
            require(ports == record['interfaces'], 'Preparation interfaces differ')
            require(estimate_jargon_score(PLAIN) < .4, 'Description needs plainer language')
            for kind, content in [('technical', CONTRACT), ('dejargonized', PLAIN)]:
                keys = {key: IDENTITY, 'kind': kind, 'language': 'en'}
                ensure(prefix+'_descriptions', keys, dict(keys, content=content, generated_by=RUNNER,
                    reviewed=False, jargon_score=round(estimate_jargon_score(content), 6)))
            keys = {key: IDENTITY, 'version_id': version, 'name': 'payload'}
            ensure(prefix+'_parameters', keys, dict(keys, type_desc='dict', required=True,
                default_value_repr='', position=0, kind='positional_or_keyword', technical_description=CONTRACT,
                dejargonized_description='Caller supplies private series history, group roles, dates, known covariates and training start days.',
                constraints_json=Jsonb(dict(parameter_contract=CONTRACT, validity_bounds=CONTRACT,
                    review_status='automated_tier_3_pass_with_limits'))))
        evidence = uuid5(version, RUNNER)
        ensure('artifact_audit_evidence', {'evidence_id': evidence}, dict(evidence_id=evidence, artifact_id=IDENTITY,
            version_id=version, audit_type='semantic_audit', passed=True, status='completed', source_kind='automated',
            runner_version=RUNNER, details=Jsonb(dict(scope='Missing preparation descriptions and parameter metadata only; existing approval, interfaces and prior conditions preserved.',
                implementation_sha256=record['provider_sha256'], publication_tier=3, review_sha256=sha(path),
                repair_script_sha256=sha(Path(__file__))))))
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            require(db.execute(f'SELECT {prefix}_is_publishable(%s) AS eligible', (IDENTITY,)).fetchone()['eligible'], 'Native preparation eligibility not restored')
            require(db.execute(f'SELECT count(DISTINCT {key}) AS n FROM catalog_{prefix}s_served WHERE {key}=%s', (IDENTITY,)).fetchone()['n'] == 1, 'Preparation provider not served')
        require(protected(db) == before, 'Prior history, approval or other records changed')
        require(review() == qualification, 'Qualification changed during repair')
        require(fail_after is None, 'Failure point not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply, rows_created=created, native_eligible_providers=1,
                existing_approval_and_history_preserved=True, new_provider_approvals=0,
                script_sha256=sha(Path(__file__)), review_sha256=sha(path))


def validate():
    with connect() as db:
        before = snapshot(db, [IDENTITY])
        outside = protected(db)
    def unchanged():
        with connect() as db:
            require(snapshot(db, [IDENTITY]) == before and protected(db) == outside, 'Metadata repair rollback incomplete')
    dry = repair()
    unchanged()
    require(dry['rows_created'] == 7, 'Expected seven missing metadata/evidence rows')
    for point in (1, 4, 7):
        try:
            repair(fail_after=point)
        except InjectedFailure:
            pass
        else:
            raise ValueError('Injected failure not reached')
        unchanged()
    return dict(passed=True, all_rollbacks_complete=True, catalog_mutations=0, rollback_points=[1, 4, 7],
                dry_run=dry, script_sha256=sha(Path(__file__)), review_sha256=dry['review_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'apply', 'repeat'])
    action = parser.parse_args().action
    result = validate() if action == 'validate' else repair(apply=True)
    if action == 'repeat':
        require(result['rows_created'] == 0, 'Repeat repair changed rows')
    destination = GATES if action == 'validate' else ROOT/f'docs/reviews/m5_preparation_metadata_{action}.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
