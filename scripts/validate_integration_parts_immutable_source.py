#!/usr/bin/env python3
"""Fresh read-only source audit for the C1 integration-by-parts reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.physics_ingest.pdg_evidence import _digest, prepare_pdg_evidence
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars
from sciona.ghost.dimensions import DimensionalSignature
from sciona.physics_ingest.integration_parts_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof


EXPECTED = {
    '8489593958': ('d(u v)', 'u dv + v du'),
    '8489593960': ('d(u v) - v du', 'u dv'),
    '8489593962': ('u dv', 'd(u v) - v du'),
    '8489593964': (r'\int u dv', r'u v - \int v du'),
}


def validate(root, symbol_file, rule_file):
    symbol_bytes = symbol_file.read_bytes()
    records = {}
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        graph = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash']!=SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes = db.execute('SELECT node_id,name FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (SOURCE_VERSION,)).fetchall()
        expected_rules = ['subtract X from both sides', 'swap LHS with RHS', 'indefinite integration']
        if [n['name'] for n in nodes] != expected_rules:
            raise ValueError('Source inference flow differs')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(bindings) != 6 or any(b['status'] != 'active' for b in bindings):
            raise ValueError('Source binding inventory differs')
        rule_pins = set()
        for binding in bindings:
            rows = db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s', (binding['bound_artifact_fqdn'], binding['bound_version_content_hash'])).fetchall()
            if len(rows) != 1:
                raise ValueError('Unique source expression required')
            row = rows[0]
            payload = row['source_payload']
            identity = payload['id']
            raw = payload['raw_payload']
            sides = tuple(raw[key].replace('\\\\', '\\') for key in ['latex_lhs', 'latex_rhs'])
            if identity not in EXPECTED or sides != EXPECTED[identity]:
                raise ValueError('Reviewed source equation differs: '+identity)
            evidence = prepare_pdg_evidence(row, symbol_bytes)
            definitions=load_pinned_pdg_scalars(symbol_bytes,evidence['symbol_file_sha256'])
            for source_id,name in [('pdg0004221','u'),('pdg0005177','v')]:
                if definitions[source_id].latex!=name or definitions[source_id].dimension!=DimensionalSignature():
                    raise ValueError('Source scalar identity or dimension changed')
            rule_pins.add(row['snapshot_payload']['core_file_sha256']['conversion_of_data_formats/infrules.cypher'])
            records[identity] = dict(source_payload_sha256=_digest(payload),
                candidate_payload_sha256=evidence['candidate_payload_sha256'], fresh_source_evidence_sha256=_digest(evidence),
                source_equation={key: raw.get(key) for key in ['latex_lhs', 'latex_rhs', 'sympy_lhs', 'sympy_rhs']})
        if set(records) != set(EXPECTED) or len(rule_pins) != 1:
            raise ValueError('Four source equations and one inference pin required')
        load_pinned_rule_contracts(rule_file.read_bytes(), next(iter(rule_pins)))
    result=dict(approved=False, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_nodes=3, source_bindings=6, source_records=records,
                symbol_sha256=hashlib.sha256(symbol_bytes).hexdigest(), rule_sha256=next(iter(rule_pins)),
                proof=verify_proof(build_proof()),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                proof_sha256=hashlib.sha256((root/'sciona/physics_ingest/integration_parts_proof.py').read_bytes()).hexdigest(),
                scope='Fresh source audit and explicit parametrized differential reconstruction; not literal source AST parity or publication approval.')
    historical=json.loads((root/'docs/reviews/integration_parts_execution.json').read_text())['source_proof']
    if historical['validator_sha256']!=hashlib.sha256((root/'scripts/validate_integration_parts_source.py').read_bytes()).hexdigest():
        raise ValueError('Historical source validator changed')
    if {k:v for k,v in result.items() if k!='validator_sha256'}!={k:v for k,v in historical.items() if k!='validator_sha256'}:
        raise ValueError('Immutable source qualification differs')
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file', 'rule-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.symbol_file, args.rule_file)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['source_nodes', 'source_bindings', 'scope']}))
