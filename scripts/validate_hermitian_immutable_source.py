#!/usr/bin/env python3
"""Fresh read-only source audit for the finite-matrix adjoint reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.physics_ingest.pdg_evidence import _digest, prepare_pdg_evidence
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.hermitian_expectation_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof


EXPECTED = {
    '9999999975': (r'\langle \psi| \hat{A} |\psi \rangle', r'\langle a \rangle'),
    '9294858532': (r'\hat{A}^+', r'\hat{A}'),
    '2394935835': (r'\left(\langle\psi| \hat{A} |\psi \rangle \right)^+', r'\left(\langle a \rangle\right)^+'),
    '1010393913': (r'\langle \psi| \hat{A}^+ |\psi \rangle', r'\langle a \rangle^*'),
    '4948934890': (r'\langle \psi| \hat{A} |\psi \rangle', r'\langle a \rangle^*'),
    '2848934890': (r'\langle a \rangle^*', r'\langle a \rangle'),
}


def validate(root, symbol_file, rule_file):
    symbol_bytes = symbol_file.read_bytes()
    records = {}
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        graph = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Expected original draft differs')
        nodes = db.execute('SELECT node_id,name FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (SOURCE_VERSION,)).fetchall()
        expected_rules = ['conjugate transpose both sides', 'distribute conjugate transpose to factors',
                          'substitute RHS of expr 1 into expr 2', 'substitute RHS of expr 1 into expr 2']
        if [n['name'] for n in nodes] != expected_rules:
            raise ValueError('Source inference flow differs')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(bindings) != 10 or any(b['status'] != 'active' for b in bindings):
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
            rule_pins.add(row['snapshot_payload']['core_file_sha256']['conversion_of_data_formats/infrules.cypher'])
            records[identity] = dict(source_payload_sha256=_digest(payload),
                candidate_payload_sha256=evidence['candidate_payload_sha256'], fresh_source_evidence_sha256=_digest(evidence),
                source_equation={key: raw.get(key) for key in ['latex_lhs', 'latex_rhs', 'sympy_lhs', 'sympy_rhs']})
        if set(records) != set(EXPECTED) or len(rule_pins) != 1:
            raise ValueError('Six source equations and one inference pin required')
        load_pinned_rule_contracts(rule_file.read_bytes(), next(iter(rule_pins)))
    result=dict(approved=False, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_nodes=4, source_bindings=10, source_records=records,
                symbol_sha256=hashlib.sha256(symbol_bytes).hexdigest(), rule_sha256=next(iter(rule_pins)),
                proof=verify_proof(build_proof()),
                validator_sha256=hashlib.sha256((root/'scripts/validate_hermitian_expectation_source.py').read_bytes()).hexdigest(),
                proof_sha256=hashlib.sha256((root/'sciona/physics_ingest/hermitian_expectation_proof.py').read_bytes()).hexdigest(),
                scope='Fresh source audit and explicit finite-dimensional reconstruction; not literal source AST parity or publication approval.')

    historical=json.loads((root/'docs/reviews/hermitian_execution.json').read_text())['source_proof']
    if result!=historical:raise ValueError('Immutable source qualification differs')
    result['immutable_validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file', 'rule-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.symbol_file, args.rule_file)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['source_nodes', 'source_bindings', 'scope']}))
