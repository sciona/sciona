"""Restore missing metadata for two unchanged, already approved providers."""
import argparse
import inspect
import json
from pathlib import Path
from uuid import UUID, uuid5

from psycopg.types.json import Jsonb
from sciona.atoms.ml import aptos_execution as provider
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.review_decomposed_aptos_baseline import review, require, ROOT, sha, build_graph
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

RUNNER = 'aptos-approved-provider-metadata-repair.v1'
PLAN = ROOT/'docs/reviews/competition_aptos_publication_plan.json'
IDENTITIES = [UUID('68b983ad-1ea6-5c16-abb4-d8bedd8d0670'), UUID('7af30c0b-05e9-5bb2-bb73-e46cc49e73d1')]
PLAIN = {
    'aptos_population': 'Check four separate groups before training. Keep supplied labels and unlabeled examples in their assigned roles. Require unique identifiers with no overlap between groups. This checks identifiers, not whether two images show the same content. No model is trained here.',
    'aptos_train_ensemble': 'Train eight image models, combine their predictions to form added training targets, then train each model for ten more epochs. Continue each model from its own saved weights. Check saved weights by replay and remove temporary files. Return scores, classes and training histories. This is a source-based reference; its prediction quality is not established.',
}
CONTRACT = {
    'aptos_population': 'Validate four nonempty, identity-disjoint source roles. Base and averaging supplied labels are finite integers 0..4; grouped labels are integers 0..3. Labeled roles are pairs of unique nonempty string keys and aligned finite numeric label vectors. Pseudo keys have no supplied labels. Return a Population with read-only label arrays and fixed role ordering. Caller establishes role provenance and physical-image identity; identity disjointness alone does not detect duplicate content.',
    'aptos_train_ensemble': 'Revalidate a Population instance and runtime image coverage. Query keys must be nonempty and unique, disjoint from supplied-label roles; overlap with pseudo keys is allowed. Require all eight family/replica configurations, positive five-epoch increments for stage one, distinct integer seeds in [0,2**32), a positive integer batch size and finite positive learning rate. Load all four pinned caller-managed safetensors backbones, train at native family resolutions with trainable GeM, scalar regression heads and Smooth-L1 loss. Train eight first-stage models before forming the arithmetic ensemble teacher; preserve base labels, average the five-level role, bound the four-level role around teacher group means and retain soft pseudo targets. Continue each own checkpoint for ten epochs with fresh Adam and the reference augmentation, retaining partial batches. Verify all saved checkpoints by exact prediction replay and remove temporary files on exit. Return scores, thresholded ordinals, all sixteen fit histories, branch predictions and soft targets. Threshold equality and group bounds are explicit caller policies. This is the qualified CPU reconstruction, not historical winning-runtime or competitive-accuracy equivalence. Training changes Torch RNG state; no RNG preservation guarantee is made.',
}


class InjectedFailure(RuntimeError):
    pass


def protected(db):
    state = {}
    for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
        for suffix in ('s', '_versions', '_io_specs', '_references', '_audit_rollups'):
            table = prefix+suffix
            expression = "to_jsonb(t)-'updated_at'" if suffix == 's' else 'to_jsonb(t)'
            state[table] = comparable(db.execute(f'SELECT {expression} AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY to_jsonb(t)::text', (IDENTITIES,)).fetchall())
    state['prior_evidence'] = comparable(db.execute('''SELECT to_jsonb(t) AS row FROM artifact_audit_evidence t
        WHERE artifact_id=ANY(%s) AND runner_version IS DISTINCT FROM %s ORDER BY to_jsonb(t)::text''', (IDENTITIES, RUNNER)).fetchall())
    return state


def repair(apply=False, fail_after=None):
    reviewed = review()
    frozen = json.loads((ROOT/'docs/reviews/decomposed_aptos_baseline_review.json').read_text())
    require(reviewed == frozen, 'Current baseline review changed')
    plan = json.loads(PLAN.read_text())
    records = plan['providers']
    require([UUID(r['artifact_id']) for r in records] == IDENTITIES, 'Provider identities differ')
    nodes = {n.matched_primitive: n for n in build_graph().nodes}
    directory = ROOT.parent/'sciona-atoms-ml'
    specs = {spec.fqdn: spec for spec in _parse_registered_atoms(
        repo=ProviderRepo('sciona-atoms-ml', directory), artifact_root=directory/'src/sciona/atoms')
        if spec.fqdn in nodes}
    require(set(specs) == set(nodes), 'Exact provider registrations missing')
    writes = 0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        before = protected(db)
        def ensure(table, keys, row):
            nonlocal writes
            writes += int(ensure_row(db, table, keys, row))
            if fail_after == writes:
                raise InjectedFailure('Injected metadata repair failure')
        for record in records:
            identity, version = UUID(record['artifact_id']), UUID(record['version_id'])
            symbol = record['source_symbol']
            function = getattr(provider, symbol)
            spec = specs[record['fqdn']]
            require(str(spec.version_id) == str(version) and spec.content_hash == record['content_hash'],
                    'Provider registration fingerprint differs')
            require(sha(Path(inspect.getfile(function))) == record['source_sha256'] == reviewed['provider_sha256'], 'Provider source differs')
            node = nodes[record['fqdn']]
            signature = inspect.signature(function)
            require(list(signature.parameters) == [p.name for p in node.inputs], 'Provider parameter order differs')
            ports = [dict(direction=d, name=p.name, ordinal=i, type_desc=p.type_desc, required=p.required,
                default_value_repr=p.default_value_repr or '') for d in ('input', 'output')
                for i, p in enumerate(getattr(node, d+'s'))]
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                actual = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                    WHERE a.{key}=%s AND v.is_latest FOR UPDATE OF a,v''', (identity,)).fetchall()
                require(actual == [dict(fqdn=record['fqdn'], status='approved', is_publishable=True,
                    version_id=str(version), content_hash=record['content_hash'], trust_tier=3)], 'Approved provider version differs')
                actual_ports = db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
                require(actual_ports == ports, 'Approved provider interface differs')
                require(estimate_jargon_score(PLAIN[symbol]) < .4, 'Description needs plainer language')
                for kind, content in [('technical', CONTRACT[symbol]), ('dejargonized', PLAIN[symbol])]:
                    keys = {key: identity, 'kind': kind, 'language': 'en'}
                    ensure(prefix+'_descriptions', keys, dict(keys, content=content, generated_by=RUNNER,
                        reviewed=False, jargon_score=round(estimate_jargon_score(content), 6)))
                for i, port in enumerate(node.inputs):
                    parameter = signature.parameters[port.name]
                    keys = {key: identity, 'version_id': version, 'name': port.name}
                    ensure(prefix+'_parameters', keys, dict(keys, type_desc=port.type_desc, required=port.required,
                        default_value_repr=port.default_value_repr or '', position=i, kind=parameter.kind.name.lower(),
                        technical_description=port.constraints,
                        dejargonized_description='Caller supplies '+port.name.replace('_', ' ')+'.',
                        constraints_json=Jsonb(dict(parameter_contract=port.constraints, validity_bounds=CONTRACT[symbol],
                            review_status='automated_tier_3_pass_with_limits'))))
            evidence = uuid5(version, RUNNER)
            ensure('artifact_audit_evidence', {'evidence_id': evidence}, dict(evidence_id=evidence, artifact_id=identity,
                version_id=version, audit_type='semantic_audit', passed=True, status='completed', source_kind='automated',
                runner_version=RUNNER, details=Jsonb(dict(scope='Missing description and parameter metadata only; prior approval unchanged.',
                    implementation_sha256=reviewed['provider_sha256'], publication_tier=3,
                    review_sha256=sha(ROOT/'docs/reviews/decomposed_aptos_baseline_review.json'),
                    repair_script_sha256=sha(ROOT/'scripts/repair_decomposed_aptos_metadata.py')))))
            eligible = db.execute('SELECT atom_is_publishable(%s) AS atom,artifact_is_publishable(%s) AS artifact',
                (identity, identity)).fetchone()
            require(eligible == dict(atom=True, artifact=True), 'Native eligibility not restored')
        require(protected(db) == before, 'Provider history or approval changed')
        require(review() == reviewed, 'Review evidence changed during repair')
        require(fail_after is None, 'Requested failure point not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply, rows_created=writes, native_eligible_providers=2,
        existing_approval_and_history_preserved=True, new_provider_approvals=0,
        script_sha256=sha(ROOT/'scripts/repair_decomposed_aptos_metadata.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    print(json.dumps(repair(apply=parser.parse_args().apply), indent=2))
