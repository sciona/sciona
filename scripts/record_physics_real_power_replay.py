"""Record fresh conditional replay and explicit assumptions without approval."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.physics_ingest.pdg_evidence import _digest
from scripts.import_residual_execution_drafts import ensure_row
from scripts.validate_physics_real_power_replay import validate

ROOT = Path(__file__).resolve().parents[1]
RUNNER = 'pdg-graph-replay.v3'


def record(symbol_file, rule_file, apply=False):
    reviewed = validate(ROOT, symbol_file, rule_file)
    if not reviewed['graphs'] or not all(g['passed'] for g in reviewed['graphs']):
        raise ValueError('Fresh successful replay required')
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in [('symbols', symbol_file), ('rules', rule_file)]}
    created = 0
    identities = []
    before = {}
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        for graph in reviewed['graphs']:
            version = UUID(graph['version_id'])
            target = db.execute('SELECT v.artifact_id,v.content_hash,a.status,a.is_publishable FROM artifact_versions v '
                'JOIN artifacts a USING(artifact_id) WHERE v.version_id=%s FOR SHARE OF a,v', (version,)).fetchone()
            if not target or target['content_hash'] != graph['content_hash'] or target['status'] != 'draft' or target['is_publishable']:
                raise ValueError('Target draft identity differs')
            prior = db.execute('SELECT details,passed,version_id FROM artifact_audit_evidence WHERE evidence_id=%s FOR SHARE',
                               (graph['prior_failed_evidence_id'],)).fetchone()
            if not prior or prior['passed'] or str(prior['version_id']) != str(version) or _digest(prior['details']) != graph['prior_evidence_sha256']:
                raise ValueError('Prior failed replay identity differs')
            details = dict(replay=graph, implementation_sha256=reviewed['implementation_sha256'], source_sha256=source_hashes,
                recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitations=reviewed['limitations'],
                scope='Conditional forward source-algebra replay; does not approve physical premises, runtime selection, or Tier 1 certification.')
            evidence = uuid5(version, RUNNER + ':' + _digest(details))
            before[str(evidence)] = db.execute('SELECT passed FROM artifact_audit_evidence WHERE evidence_id=%s', (evidence,)).fetchone()
            created += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence}, dict(evidence_id=evidence,
                artifact_id=target['artifact_id'], version_id=version, audit_type='determinism_replay', passed=True,
                status='completed', details=Jsonb(details), source_kind='automated', runner_version=RUNNER))
            for condition in graph['required_conditions']:
                identity = uuid5(evidence, condition)
                created += ensure_row(db, 'artifact_validity_bounds', {'bound_id': identity}, dict(bound_id=identity,
                    artifact_id=target['artifact_id'], version_id=version, scope='version', bound_kind='assumption',
                    validity_statement='Necessary algebraic condition: ' + condition, review_status='unreviewed',
                    metadata=Jsonb(dict(replay_evidence_id=str(evidence), condition_srepr=condition,
                        scope='Required premise of the replay; applicability to caller inputs remains to be checked.'))))
            identities.append(dict(version_id=str(version), evidence_id=str(evidence), conditions=len(graph['required_conditions'])))
        if validate(ROOT, symbol_file, rule_file) != reviewed:
            raise ValueError('Replay evidence changed during transaction')
        if not apply:
            db.rollback()
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for item in identities:
            evidence = db.execute('SELECT passed FROM artifact_audit_evidence WHERE evidence_id=%s', (item['evidence_id'],)).fetchone()
            if apply and evidence != {'passed': True}:
                raise ValueError('Committed replay missing')
            if not apply and evidence != before[item['evidence_id']]:
                raise ValueError('Rollback did not preserve previous evidence state')
            if apply:
                count = db.execute("SELECT count(*) AS n FROM artifact_validity_bounds WHERE metadata->>'replay_evidence_id'=%s",
                                   (item['evidence_id'],)).fetchone()['n']
                if count != item['conditions']:
                    raise ValueError('Committed assumptions missing')
    return dict(applied=apply, rows_created=created, replays=identities, approval_applied=False,
                prior_failed_evidence_retained=True, conditions_require_applicability_check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol-file', type=Path, required=True)
    parser.add_argument('--rule-file', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(record(args.symbol_file, args.rule_file, args.apply)))
