"""Restore missing metadata for the existing approved OpenVaccine preparation atom."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5

from psycopg.types.json import Jsonb
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.review_decomposed_openvaccine_implementation import review
from scripts.promote_nab_decomposed_providers import TABLES, snapshot
from scripts.review_decomposed_openvaccine_implementation import ROOT, sha, require
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

RUNNER = 'openvaccine-preparation-metadata-repair.v1'
IDENTITY = UUID('12c53ef6-e2c3-52fe-a8e1-421c39039636')
GATES = ROOT/'docs/reviews/openvaccine_preparation_metadata_gates.json'
PLAIN = 'Check private input populations, observed targets, model splits and training choices. Require valid values and a complete plan for every model and round. Return a private copy. This step does not fold inputs, train models or measure prediction quality.'
CONTRACT = 'Validate a dictionary with exactly version, seed, sequences, targets, cluster_ids, proximity_factors, splits, pseudo_sequences, prediction_sequences, pseudo_rounds, pretraining_steps, initial_supervised_steps, rollback_policy, absolute_tolerance and std_ddof. Require strict finite JSON serialization, with null for masked targets; no encoded-byte limit or integrity fingerprint is implemented. Version is Python integer one excluding bool. Seed is Python integer in [0,2**31); both step counts are positive Python integers. Each of the three populations is nonempty, contains only ACGU strings of length at least two and has equal lengths within that population; lengths may differ between populations. Targets convert to float32 with shape (labeled_count,labeled_length,5), allow NaN masks from null, reject infinity and require usable observed targets in every split. Numeric payload arrays accept Python int/float excluding bool, with null allowed only for masked targets. Cluster membership is a Python integer list. Float32 proximity factors are aligned, finite, nonnegative and have positive support; population weights divide each factor by square root of full-population cluster size and must retain finite positive total weight. Require exactly one split for each of twenty family/slot identities: five slots each of gru, lstm, forward and wave. Train and validation indices are nonempty distinct in-range Python integers, disjoint within each member, and each partition has observed targets with positive weight. This does not establish cluster or sequence independence. Every nonempty refinement round supplies all twenty member plans with exactly supervised_reverse_flags, eligible, maximum_uncertainty, perturbations, sample_weights and reverse_flags. Validate aligned boolean masks/reversal flags, finite float32 perturbations, nonnegative positive-total float32 pseudo weights, at least one eligible positive-weight target and finite nonnegative scalar uncertainty threshold. Rollback policy is weights_only or model_and_optimizer, tolerance is finite nonnegative and nonboolean, and std_ddof is Python integer zero or one. Return frozen Prepared wrapping a deep-copied payload dictionary; the nested dictionary remains mutable and downstream consumers revalidate it. No folding, training, seed reset, model checkpoint loading, historical accuracy or original-CDG approval is implied. Pseudo-label uncertainty retention and subsequent numerical training checks remain downstream responsibilities.'


class InjectedFailure(RuntimeError):
    pass


def protected(db):
    # Protect the target's complete prior history and all other related records.
    plan = json.loads((ROOT/'docs/reviews/openvaccine_decomposed_provider_intake_plan.json').read_text())
    ids = [UUID(r['atom_id']) for r in plan['providers']] + [UUID(x) for x in (
        'd132e834-0634-59a6-adc5-2fe3b00082e3', '26cf554f-1920-5652-853d-99b62dc3aade',
        '19cf56e7-8fc1-5dad-ac8a-51d58c08f6e9', '04ee2813-9cda-5a2f-807d-879dec6ff290',
        '4f486f1c-82db-50db-b5d2-188aeaa8cc4a')]
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
    path = ROOT/'docs/reviews/decomposed_openvaccine_implementation_review.json'
    integration = json.loads(path.read_text())
    require(qualification == integration, 'Integration review differs')
    records = [r for r in integration['providers'] if r['runtime_fqdn'].endswith('.openvaccine_prepare')]
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
                dejargonized_description='Caller supplies private input populations, observed targets, model splits and complete training controls.',
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
    destination = GATES if action == 'validate' else ROOT/f'docs/reviews/openvaccine_preparation_metadata_{action}.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
