#!/usr/bin/env python3
"""Recover missing vacuum_wave equations from pinned public Cypher source."""
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
from sciona.physics_ingest.vacuum_wave_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof
from sciona.ghost.dimensions import DimensionalSignature

EXPECTED = {'1314464131': ('\\vec{ \\nabla} \\times \\frac{\\partial \\vec{H}}{\\partial t}', '\\epsilon_0 \\frac{\\partial^2 \\vec{E}}{\\partial t^2}'), '1314864131': ('\\vec{ \\nabla} \\times \\vec{H}', '\\epsilon_0 \\frac{\\partial }{\\partial t}\\vec{E}'), '1636453295': ('\\vec{ \\nabla} \\times \\vec{ \\nabla} \\times \\vec{E}', '- \\nabla^2 \\vec{E}'), '3947269979': ('\\vec{ \\nabla} \\times \\vec{ \\nabla} \\times \\vec{E}', '-\\mu_0 \\epsilon_0 \\frac{\\partial^2 \\vec{E}}{\\partial t^2}'), '7466829492': ('\\vec{ \\nabla} \\cdot \\vec{E}', '0'), '7575859295': ('\\vec{ \\nabla} \\times \\vec{ \\nabla} \\times \\vec{E}', '\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})'), '8494839423': ('\\nabla^2 \\vec{E}', '\\mu_0 \\epsilon_0 \\frac{\\partial^2 \\vec{E}}{\\partial t^2}'), '9291999979': ('\\vec{ \\nabla} \\times \\vec{ \\nabla} \\times \\vec{E}', '-\\mu_0\\vec{ \\nabla} \\times \\frac{\\partial \\vec{H}}{\\partial t}'), '9919999981': ('\\rho', '0'), '9991999979': ('\\vec{ \\nabla} \\times \\vec{E}', '-\\mu_0\\frac{\\partial \\vec{H}}{\\partial t}'), '9999999981': ('\\vec{ \\nabla} \\cdot \\vec{E}', '\\rho/\\epsilon_0')}


def validate(root, symbol_file, rule_file, expression_file):
    records, snapshots, pins = {}, {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        graph = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Expected original draft differs')
        nodes = db.execute('SELECT node_id,name,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (SOURCE_VERSION,)).fetchall()
        if [n['name'] for n in nodes] != ['partially differentiate with respect to','take curl of both sides']+['substitute LHS of expr 1 into expr 2']*4:
            raise ValueError('Source steps differ')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(bindings) != 16 or any(b['status'] != 'active' for b in bindings):
            raise ValueError('Source bindings differ')
        if {b['bound_artifact_fqdn'].rsplit('.', 1)[1] for b in bindings} != set(EXPECTED):
            raise ValueError('Eleven exact source identities required')
        expected_steps = [
            ('111680', {'1314864131','1314464131'}),
            ('111776', {'9991999979','9291999979'}),
            ('111556', {'9291999979','1314464131','3947269979'}),
            ('111556', {'9999999981','9919999981','7466829492'}),
            ('111556', {'7575859295','7466829492','1636453295'}),
            ('111556', {'1636453295','3947269979','8494839423'}),
        ]
        for node, (rule, identities) in zip(nodes, expected_steps):
            signature = json.loads(node['type_signature'])
            actual = {b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            if signature['inference_rule_id'] != rule or actual != identities:
                raise ValueError('Source rule or per-step bindings differ')
            feeds = signature['variable_bindings']['feeds']
            if rule == '111680':
                if len(feeds)!=1 or feeds[0]['sympy']!="Symbol('pdg0001467')" or feeds[0]['latex']!='t':
                    raise ValueError('Source differentiation must be by time')
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
        if set(snapshots) != {'1314464131','1314864131','1636453295','3947269979','7466829492','9291999979','9919999981','9991999979','9999999981'} or not pins or any(p != pins[0] for p in pins):
            raise ValueError('Expected stored source coverage differs')
    pin = pins[0]; content = expression_file.read_bytes()
    if hashlib.sha256(content).hexdigest() != pin['conversion_of_data_formats/expr_and_feed.cypher']:
        raise ValueError('Expression source file differs from original ingestion pin')
    definitions = load_pinned_pdg_scalars(symbol_file.read_bytes(), pin['conversion_of_data_formats/symbols.cypher'])
    load_pinned_rule_contracts(rule_file.read_bytes(), pin['conversion_of_data_formats/infrules.cypher'])
    scalar_expectations = {
        'pdg0001467': ('t',DimensionalSignature(T=1)),
        'pdg0002069': (r'\vec{H}',DimensionalSignature()),
        'pdg0004326': (r'\vec{E}',DimensionalSignature()),
        'pdg0006238': ('E',DimensionalSignature()),
        'pdg0006197': (r'\mu_0',DimensionalSignature(M=1,L=1,T=-2,I=-2)),
        'pdg0007940': (r'\epsilon_0',DimensionalSignature(M=-1,L=-3,T=4,I=2)),
        'pdg0003935': (r'\rho',DimensionalSignature(M=1,L=-3)),
    }
    for identity,(latex,dimension) in scalar_expectations.items():
        if definitions[identity].latex.replace(chr(92)*2,chr(92)) != latex or definitions[identity].dimension != dimension:
            raise ValueError('Reviewed source symbol identity/dimension differs')
    length,time=DimensionalSignature(L=1),DimensionalSignature(T=1)
    electric=DimensionalSignature(M=1,L=1,T=-3,I=-1)
    magnetic=DimensionalSignature(I=1,L=-1)
    charge_density=DimensionalSignature(I=1,T=1,L=-3)
    eps,mu=definitions['pdg0007940'].dimension,definitions['pdg0006197'].dimension
    dimensional_checks={
        'ampere':magnetic.divide(length)==eps.multiply(electric).divide(time),
        'faraday':electric.divide(length)==mu.multiply(magnetic).divide(time),
        'gauss':electric.divide(length)==charge_density.divide(eps),
        'wave_equation':electric.divide(length.power(2))==mu.multiply(eps).multiply(electric).divide(time.power(2)),
    }
    if not all(dimensional_checks.values()):
        raise ValueError('Corrected SI dimensions do not satisfy Maxwell equations')
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
    result=dict(approved=False, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                stored_source_equations=9, recovered_missing_equations=2, source_records=records, stored_evidence=snapshots,
                source_file_sha256={k: pin['conversion_of_data_formats/'+k+'.cypher'] for k in ['symbols','infrules','expr_and_feed']},
                corrected_proof=verify_proof(build_proof()),
                dimensional_checks=dimensional_checks,
                corrected_dimensions=dict(E=str(electric),H=str(magnetic),rho=str(charge_density)),
                corrections=['Restore componentwise vector derivatives, correct field identity and operand ordering; source ASTs are incomplete or scalar placeholders.',
                             'Repair source curl-curl LaTeX to grad(div E)-laplacian(E), with gradient acting only on divergence.',
                             'Dimensionless source E/H corrected to SI V/m and A/m; source mass-density rho corrected to charge density for Gauss law.',
                             'Make zero current, zero charge, constant vacuum coefficients and C2 Cartesian fields explicit.'],
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                    ['scripts/validate_vacuum_wave_source.py','sciona/physics_ingest/vacuum_wave_proof.py']})

    historical=json.loads((root/'docs/reviews/vacuum_wave_execution.json').read_text())['source_proof']
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
