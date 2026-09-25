#!/usr/bin/env python3
"""Recover missing sine_double_angle equations from pinned public Cypher source."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest
from sciona.physics_ingest.sine_double_angle_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof, verify_source_equations
from sciona.ghost.dimensions import DimensionalSignature

EXPECTED = {
    '2103023049': (r'\sin(x)', r'\frac{1}{2i}\left(\exp(i x)-\exp(-i x) \right)'),
    '4585932229': (r'\cos(x)', r'\frac{1}{2}\left(\exp(i x)+\exp(-i x) \right)'),
    '8483686863': (r'\sin(2 x)', r'\frac{1}{2i}\left(\exp(i 2 x)-\exp(-i 2 x) \right)'),
    '3470587782': (r'\sin(x) \cos(x)', r'\frac{1}{2i}\left(\exp(i x)-\exp(-i x) \right) \frac{1}{2}\left(\exp(i x)+\exp(-i x) \right)'),
    '9894826550': (r'2 \sin(x) \cos(x)', r'\frac{1}{2i}\left(\exp(i x)-\exp(-i x) \right) \left(\exp(i x)+\exp(-i x) \right)'),
    '8699789241': (r'2 \sin(x) \cos(x)', r'\frac{1}{2 i} \left( \exp(i 2 x) - 1 + 1 - \exp(-i 2 x) \right)'),
    '9180861128': (r'2 \sin(x) \cos(x)', r'\frac{1}{2 i} \left( \exp(i 2 x) - \exp(-i 2 x) \right)'),
    '2405307372': (r'\sin(2 x)', r'2 \sin(x) \cos(x)'),
}


def validate(root, symbol_file, rule_file, expression_file):
    records, snapshots, pins = {}, {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        graph = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash']!=SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes = db.execute('SELECT node_id,name,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (SOURCE_VERSION,)).fetchall()
        if [n['name'] for n in nodes] != ['change variable X to Y','multiply expr 1 by expr 2','multiply both sides by','simplify','simplify','RHS of expr 1 equals RHS of expr 2']:
            raise ValueError('Source steps differ')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(bindings) != 14 or any(b['status'] != 'active' for b in bindings):
            raise ValueError('Source bindings differ')
        if {b['bound_artifact_fqdn'].rsplit('.', 1)[1] for b in bindings} != set(EXPECTED):
            raise ValueError('Eight exact source identities required')
        expected_steps = [
            ('111886', {'2103023049','8483686863'}, ["Symbol('pdg0001464')", "Mul(Integer(2), Symbol('pdg0001464'))"]),
            ('111253', {'2103023049','4585932229','3470587782'}, []),
            ('111182', {'3470587782','9894826550'}, ['Integer(2)']),
            ('111457', {'9894826550','8699789241'}, []),
            ('111457', {'8699789241','9180861128'}, []),
            ('111863', {'9180861128','8483686863','2405307372'}, []),
        ]
        for node, (rule, identities, feeds) in zip(nodes, expected_steps):
            signature = json.loads(node['type_signature'])
            actual = {b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            if signature['inference_rule_id'] != rule or actual != identities:
                raise ValueError('Source rule or per-step bindings differ')
            if [f['sympy'] for f in signature['variable_bindings']['feeds']] != feeds:
                raise ValueError('Source feeds differ')
        for b in bindings:
            rows = db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s', (b['bound_artifact_fqdn'], b['bound_version_content_hash'])).fetchall()
            if not rows: continue
            if len(rows) != 1: raise ValueError('Ambiguous stored source expression')
            row = rows[0]; identity = row['source_payload']['id']
            evidence = prepare_pdg_evidence(row, symbol_file.read_bytes())
            snapshots[identity] = dict(source_payload_sha256=_digest(row['source_payload']), fresh_source_evidence_sha256=_digest(evidence))
            pins.append(row['snapshot_payload']['core_file_sha256'])
        if set(snapshots) != {'3470587782','8483686863','8699789241','9180861128','9894826550'} or not pins or any(p != pins[0] for p in pins):
            raise ValueError('Expected stored source coverage differs')
    pin = pins[0]; content = expression_file.read_bytes()
    if hashlib.sha256(content).hexdigest() != pin['conversion_of_data_formats/expr_and_feed.cypher']:
        raise ValueError('Expression source file differs from original ingestion pin')
    definitions = load_pinned_pdg_scalars(symbol_file.read_bytes(), pin['conversion_of_data_formats/symbols.cypher'])
    load_pinned_rule_contracts(rule_file.read_bytes(), pin['conversion_of_data_formats/infrules.cypher'])
    from sciona.physics_ingest.euler_constant_semantics import reviewed_mapping
    mapping, semantics = reviewed_mapping(symbol_file.read_bytes(), pin['conversion_of_data_formats/symbols.cypher'])
    for block in re.split(r'(?m)^UNWIND ', content.decode()):
        match = re.match(r'\[\{id:"(\d+)"', block)
        if not match or match[1] not in EXPECTED: continue
        identity = match[1]
        if identity in records: raise ValueError('Duplicate source equation')
        fields = dict(re.findall(r'(\w+):"((?:\\.|[^"\\])*)"', block))
        sides = tuple(fields[k].replace('\\\\', '\\') for k in ['latex_lhs', 'latex_rhs'])
        if sides != EXPECTED[identity] or fields['latex_relation'] != '=':
            raise ValueError('Reviewed source equation differs: '+identity)
        records[identity] = dict(source_block_sha256=hashlib.sha256(block.encode()).hexdigest(),
                                 equation={k: fields[k] for k in ['latex_lhs', 'latex_rhs', 'sympy_lhs', 'sympy_rhs']})
    if set(records) != set(EXPECTED): raise ValueError('Missing original source equation')
    source_equations = {identity:record['equation'] for identity,record in records.items()}
    result=dict(approved=False, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                stored_source_equations=5, recovered_missing_equations=3, source_records=records, stored_evidence=snapshots,
                source_file_sha256={k: pin['conversion_of_data_formats/'+k+'.cypher'] for k in ['symbols','infrules','expr_and_feed']},
                corrected_proof=verify_proof(build_proof()),
                interpreted_source_proof=verify_source_equations(source_equations, mapping),
                constant_semantics=semantics,
                corrections=['Exact imaginary-unit identity interpreted as I despite source variable classification; no blanket name coercion.'],
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                    ['scripts/validate_sine_double_angle_source.py','sciona/physics_ingest/sine_double_angle_proof.py','sciona/physics_ingest/euler_constant_semantics.py','sciona/physics_ingest/source_symbolic.py']})
    historical=json.loads((root/'docs/reviews/sine_double_angle_execution.json').read_text())['source_proof']
    for name,digest in historical['implementation_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Historical source implementation changed')
    if result!=historical:
        raise ValueError('Immutable source qualification differs')
    result['immutable_validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.symbol_file, args.rule_file, args.expression_file)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['stored_source_equations','recovered_missing_equations','corrections']}))
