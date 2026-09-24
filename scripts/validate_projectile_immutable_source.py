#!/usr/bin/env python3
"""Validate immutable projectile source version and recover equations from pinned public Cypher."""
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
from sciona.physics_ingest.projectile_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof
from sciona.ghost.dimensions import DimensionalSignature

EXPECTED = {
    '9882526611': ('v_{0, x} t', 'x - x_0'),
    '1405465835': ('y', r'- \frac{1}{2} g t^2 + v_{0, y} t + y_0'),
    '3274926090': ('t', r'\frac{x - x_0}{v_{0, x}}'),
    '7354529102': ('y', r'- \frac{1}{2} g \left( \frac{x - x_0}{v_{0, x}} \right)^2 + v_{0, y} \frac{x - x_0}{v_{0, x}} + y_0'),
}


def validate(root, symbol_file, rule_file, expression_file):
    records, snapshots, pins = {}, {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        graph = db.execute('SELECT a.artifact_id::text,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if graph != dict(artifact_id='99ea69f3-a28c-51b7-8749-187b5675e7fe', content_hash=SOURCE_HASH):
            raise ValueError('Pinned original source differs')
        nodes = db.execute('SELECT node_id,name,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (SOURCE_VERSION,)).fetchall()
        if [n['name'] for n in nodes] != ['divide both sides by', 'substitute LHS of expr 1 into expr 2']:
            raise ValueError('Source steps differ')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(bindings) != 5 or any(b['status'] != 'active' for b in bindings):
            raise ValueError('Source bindings differ')
        if {b['bound_artifact_fqdn'].rsplit('.', 1)[1] for b in bindings} != set(EXPECTED):
            raise ValueError('Four exact source identities required')
        for b in bindings:
            rows = db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s', (b['bound_artifact_fqdn'], b['bound_version_content_hash'])).fetchall()
            if not rows: continue
            if len(rows) != 1: raise ValueError('Ambiguous stored source expression')
            row = rows[0]; identity = row['source_payload']['id']
            evidence = prepare_pdg_evidence(row, symbol_file.read_bytes())
            snapshots[identity] = dict(source_payload_sha256=_digest(row['source_payload']), fresh_source_evidence_sha256=_digest(evidence))
            pins.append(row['snapshot_payload']['core_file_sha256'])
        if set(snapshots) != {'3274926090', '7354529102'} or not pins or any(p != pins[0] for p in pins):
            raise ValueError('Expected stored source coverage differs')
    pin = pins[0]; content = expression_file.read_bytes()
    if hashlib.sha256(content).hexdigest() != pin['conversion_of_data_formats/expr_and_feed.cypher']:
        raise ValueError('Expression source file differs from original ingestion pin')
    definitions = load_pinned_pdg_scalars(symbol_file.read_bytes(), pin['conversion_of_data_formats/symbols.cypher'])
    load_pinned_rule_contracts(rule_file.read_bytes(), pin['conversion_of_data_formats/infrules.cypher'])
    scalar_expectations = {
        'pdg0001467': ('t', DimensionalSignature(T=1)), 'pdg0002958': ('v_{0, x}', DimensionalSignature(L=1,T=-1)),
        'pdg0001572': ('x_0', DimensionalSignature(L=1)), 'pdg0004037': ('x', DimensionalSignature(L=1)),
        'pdg0005647': ('y', DimensionalSignature(L=1)), 'pdg0001469': ('y_0', DimensionalSignature(L=1)),
        'pdg0001649': ('g', DimensionalSignature(L=1,T=-2)), 'pdg0009107': ('v_y', DimensionalSignature(L=1,T=-1)),
        'pdg0009431': ('v_{0, y}', DimensionalSignature(L=1,T=-1)),
    }
    for identity, (latex, dimension) in scalar_expectations.items():
        if definitions[identity].latex != latex or definitions[identity].dimension != dimension:
            raise ValueError('Reviewed source symbol identity/dimension differs')
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
    if "Symbol('pdg0009107')" not in records['1405465835']['equation']['sympy_rhs']:
        raise ValueError('Expected incorrect velocity identity changed')
    if "Pow(Symbol('pdg0001649'), Integer(2))" not in records['7354529102']['equation']['sympy_rhs']:
        raise ValueError('Expected incorrect gravity power changed')
    return dict(approved=False, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                stored_source_equations=2, recovered_missing_equations=2, source_records=records, stored_evidence=snapshots,
                source_file_sha256={k: pin['conversion_of_data_formats/'+k+'.cypher'] for k in ['symbols','infrules','expr_and_feed']},
                corrected_proof=verify_proof(build_proof()),
                corrections=['Instantaneous vertical-velocity symbol corrected to the explicitly stated initial vertical velocity.',
                             'Dimensionally invalid squared gravity corrected to linear gravity.'],
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                    ['scripts/validate_projectile_immutable_source.py','sciona/physics_ingest/projectile_proof.py']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.symbol_file, args.rule_file, args.expression_file)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['stored_source_equations','recovered_missing_equations','corrections']}))
