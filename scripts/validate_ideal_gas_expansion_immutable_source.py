#!/usr/bin/env python3
"""Recover missing ideal_gas_expansion equations from pinned public Cypher source."""
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
from sciona.physics_ingest.ideal_gas_expansion_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof
from sciona.ghost.dimensions import DimensionalSignature

EXPECTED = {
    '3497828859': ('V', r'\frac{n R T}{P}'),
    '3464107376': (r'\alpha', r'\frac{1}{V} \left( \frac{\partial V}{\partial T} \right)_p'),
    '1311403394': (r'\alpha', r'\frac{1}{V} \frac{nR}{P} \left( \frac{\partial T}{\partial T} \right)_P'),
    '5962145508': (r'\alpha', r'\frac{nR}{VP}'),
    '8435841627': ('P V', 'n R T'),
    '2613006036': (r'\frac{PV}{T}', 'nR'),
    '6925244346': (r'\alpha', r'\frac{PV}{T} \frac{1}{VP}'),
    '2472653783': (r'\alpha', r'\frac{1}{T}'),
}


def validate(root, symbol_file, rule_file, expression_file):
    records, snapshots, pins = {}, {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        graph = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes = db.execute('SELECT node_id,name,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (SOURCE_VERSION,)).fetchall()
        if [n['name'] for n in nodes] != ['substitute LHS of expr 1 into expr 2', 'simplify', 'divide both sides by', 'substitute RHS of expr 1 into expr 2', 'simplify']:
            raise ValueError('Source steps differ')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(bindings) != 12 or any(b['status'] != 'active' for b in bindings):
            raise ValueError('Source bindings differ')
        if {b['bound_artifact_fqdn'].rsplit('.', 1)[1] for b in bindings} != set(EXPECTED):
            raise ValueError('Eight exact source identities required')
        expected_steps = [
            ('111556', {'3497828859','3464107376','1311403394'}),
            ('111457', {'1311403394','5962145508'}),
            ('111975', {'8435841627','2613006036'}),
            ('111634', {'2613006036','5962145508','6925244346'}),
            ('111457', {'6925244346','2472653783'}),
        ]
        for node, (rule, identities) in zip(nodes, expected_steps):
            signature = json.loads(node['type_signature'])
            actual = {b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            if signature['inference_rule_id'] != rule or actual != identities:
                raise ValueError('Source rule or per-step bindings differ')
            feeds = signature['variable_bindings']['feeds']
            if rule == '111975':
                if len(feeds)!=1 or feeds[0]['sympy']!="Symbol('pdg0007343')" or feeds[0]['latex']!='T':
                    raise ValueError('Source division must be by temperature')
            elif feeds:
                raise ValueError('Unexpected feed')
        for b in bindings:
            rows = db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s', (b['bound_artifact_fqdn'], b['bound_version_content_hash'])).fetchall()
            if not rows: continue
            if len(rows) != 1: raise ValueError('Ambiguous stored source expression')
            row = rows[0]; identity = row['source_payload']['id']
            evidence = prepare_pdg_evidence(row, symbol_file.read_bytes())
            snapshots[identity] = dict(source_payload_sha256=_digest(row['source_payload']), fresh_source_evidence_sha256=_digest(evidence))
            pins.append(row['snapshot_payload']['core_file_sha256'])
        if set(snapshots) != {'1311403394','5962145508','2613006036','6925244346','2472653783'} or not pins or any(p != pins[0] for p in pins):
            raise ValueError('Expected stored source coverage differs')
    pin = pins[0]; content = expression_file.read_bytes()
    if hashlib.sha256(content).hexdigest() != pin['conversion_of_data_formats/expr_and_feed.cypher']:
        raise ValueError('Expression source file differs from original ingestion pin')
    definitions = load_pinned_pdg_scalars(symbol_file.read_bytes(), pin['conversion_of_data_formats/symbols.cypher'])
    load_pinned_rule_contracts(rule_file.read_bytes(), pin['conversion_of_data_formats/infrules.cypher'])
    scalar_expectations = {
        'pdg0004686': (r'\alpha', DimensionalSignature(Theta=-1)),
        'pdg0007343': ('T', DimensionalSignature(Theta=1)),
        'pdg0007586': ('V', DimensionalSignature(L=3)),
        'pdg0008134': ('P', DimensionalSignature(M=1,L=-1,T=-2)),
        'pdg0008179': ('R', DimensionalSignature(M=1,L=2,T=-2,Theta=-1,N=-1)),
        'pdg0002834': ('n', DimensionalSignature()),
    }
    for identity, (latex, dimension) in scalar_expectations.items():
        if definitions[identity].latex.replace('\\\\', '\\') != latex or definitions[identity].dimension != dimension:
            raise ValueError('Reviewed source symbol identity/dimension differs')
    corrected_amount = definitions['pdg0008134'].dimension.multiply(definitions['pdg0007586'].dimension).divide(
        definitions['pdg0008179'].dimension.multiply(definitions['pdg0007343'].dimension))
    if corrected_amount != DimensionalSignature(N=1):
        raise ValueError('Molar gas law must imply amount dimension')
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
    if records['6925244346']['equation']['sympy_rhs'] != "Mul(Pow(Symbol('pdg0007343'), Integer(-1)), Symbol('pdg0007586'), Symbol('pdg0008134'))":
        raise ValueError('Expected missing denominator AST changed')
    result=dict(approved=False, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                stored_source_equations=5, recovered_missing_equations=3, source_records=records, stored_evidence=snapshots,
                source_file_sha256={k: pin['conversion_of_data_formats/'+k+'.cypher'] for k in ['symbols','infrules','expr_and_feed']},
                corrected_proof=verify_proof(build_proof()),
                corrections=['Source derivative placeholders reconstructed as derivatives along a fixed-pressure path.',
                             'Intermediate 6925244346 AST omits the 1/(VP) factor explicitly present in its LaTeX.',
                             'Source n is dimensionless but its molar R requires amount dimension N1; corrected realization uses moles.'],
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                    ['scripts/validate_ideal_gas_expansion_source.py','sciona/physics_ingest/ideal_gas_expansion_proof.py']})

    historical=json.loads((root/'docs/reviews/ideal_gas_expansion_execution.json').read_text())['source_proof']
    for name,digest in historical['implementation_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Historical source implementation changed')
    if result!=historical:raise ValueError('Immutable source qualification differs')
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
