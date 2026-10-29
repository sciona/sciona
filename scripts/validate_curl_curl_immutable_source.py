#!/usr/bin/env python3
"""Recover missing curl_curl equations from pinned public Cypher source."""
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
from sciona.physics_ingest.curl_curl_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof
from sciona.ghost.dimensions import DimensionalSignature

EXPECTED = {'7575859295': ('\\vec{ \\nabla} \\times \\vec{ \\nabla} \\times \\vec{E}', '\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})'), '7575859300': ('\\epsilon^{i,j,k} \\hat{x}_i \\nabla_j ( \\vec{ \\nabla} \\times \\vec{E} )_k', '\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})'), '7575859302': ('\\epsilon^{i,j,k} \\epsilon_{n,j,k} \\hat{x}_i \\nabla_j \\nabla^m E^n', '\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})'), '7575859304': ('\\epsilon^{i,j,k} \\epsilon_{n,j,k}', '\\delta^{l}_{\\ \\ j} \\delta^{m}_{\\ \\ k} - \\delta^{l}_{\\ \\ k} \\delta^{m}_{\\ \\ h}'), '7575859306': ('\\left( \\delta^{l}_{\\ \\ j} \\delta^{m}_{\\ \\ k} - \\delta^{l}_{\\ \\ k} \\delta^{m}_{\\ \\ h} \\right) \\hat{x}_i \\nabla_j \\nabla^m E^n', '\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})'), '7575859308': ('\\left( \\delta^{l}_{\\ \\ j} \\delta^{m}_{\\ \\ k} \\hat{x}_i \\nabla_j \\nabla^m E^n\\right)-\\left( \\delta^{l}_{\\ \\ k} \\delta^{m}_{\\ \\ h} \\hat{x}_i \\nabla_j \\nabla^m E^n \\right)', '\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})'), '7575859310': ('\\hat{x}_m \\nabla_n \\nabla^m E^n - \\hat{x}_n \\nabla_m \\nabla^m E^n', '\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})'), '7575859312': ('\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})', '\\vec{ \\nabla}( \\vec{ \\nabla} \\cdot \\vec{E} - \\nabla^2 \\vec{E})')}


def validate(root, symbol_file, rule_file, expression_file):
    records, snapshots, pins = {}, {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        graph = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Expected original draft differs')
        nodes = db.execute('SELECT node_id,name,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id', (SOURCE_VERSION,)).fetchall()
        if [n['name'] for n in nodes] != ['replace curl with LeviCevita summation contravariant']*2+['substitute RHS of expr 1 into expr 2','simplify','simplify','replace summation notation with vector notation']:
            raise ValueError('Source steps differ')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(bindings) != 13 or any(b['status'] != 'active' for b in bindings):
            raise ValueError('Source bindings differ')
        if {b['bound_artifact_fqdn'].rsplit('.', 1)[1] for b in bindings} != set(EXPECTED):
            raise ValueError('Eight exact source identities required')
        expected_steps = [
            ('111935',{'7575859295','7575859300'}),
            ('111935',{'7575859300','7575859302'}),
            ('111634',{'7575859304','7575859302','7575859306'}),
            ('111457',{'7575859306','7575859308'}),
            ('111457',{'7575859308','7575859310'}),
            ('111894',{'7575859310','7575859312'}),
        ]
        for node,(rule,identities) in zip(nodes,expected_steps):
            signature=json.loads(node['type_signature'])
            actual={b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            if signature['inference_rule_id']!=rule or actual!=identities or signature['variable_bindings']['feeds']:
                raise ValueError('Per-step rule, bindings or feeds differ')
        for b in bindings:
            rows = db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s', (b['bound_artifact_fqdn'], b['bound_version_content_hash'])).fetchall()
            if not rows: continue
            if len(rows) != 1: raise ValueError('Ambiguous stored source expression')
            row = rows[0]; identity = row['source_payload']['id']
            evidence = prepare_pdg_evidence(row, symbol_file.read_bytes())
            snapshots[identity] = dict(source_payload_sha256=_digest(row['source_payload']), fresh_source_evidence_sha256=_digest(evidence))
            pins.append(row['snapshot_payload']['core_file_sha256'])
        if set(snapshots) != set(EXPECTED) or not pins or any(p != pins[0] for p in pins):
            raise ValueError('Expected stored source coverage differs')
    pin = pins[0]; content = expression_file.read_bytes()
    if hashlib.sha256(content).hexdigest() != pin['conversion_of_data_formats/expr_and_feed.cypher']:
        raise ValueError('Expression source file differs from original ingestion pin')
    definitions = load_pinned_pdg_scalars(symbol_file.read_bytes(), pin['conversion_of_data_formats/symbols.cypher'])
    load_pinned_rule_contracts(rule_file.read_bytes(), pin['conversion_of_data_formats/infrules.cypher'])
    for identity,latex in [('pdg0004326',chr(92)+'vec{E}'),('pdg0006238','E')]:
        if definitions[identity].latex.replace(chr(92)*2,chr(92))!=latex or definitions[identity].dimension!=DimensionalSignature():
            raise ValueError('Source electric-field symbol declaration differs')
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
                stored_source_equations=8, recovered_missing_equations=0, source_records=records, stored_evidence=snapshots,
                source_file_sha256={k: pin['conversion_of_data_formats/'+k+'.cypher'] for k in ['symbols','infrules','expr_and_feed']},
                corrected_proof=verify_proof(build_proof()),
                corrections=['Replace invalid repeated-index contraction with sum_k epsilon(i,j,k)*epsilon(k,m,n).',
                             'Use delta(i,m)*delta(j,n)-delta(i,n)*delta(j,m); original delta expression has unmatched indices.',
                             'Repair gradient parentheses to grad(div E)-laplacian(E); retain divergence term.',
                             'Reconstruct component derivatives from malformed scalar/nested-equality ASTs; no literal source parity.',
                             'Fix right-handed orthonormal Cartesian frame and C2 mixed-partial commutation assumptions.'],
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                    ['scripts/validate_curl_curl_source.py','sciona/physics_ingest/curl_curl_proof.py','sciona/physics_ingest/vacuum_wave_proof.py']})

    historical=json.loads((root/'docs/reviews/curl_curl_execution.json').read_text())['source_proof']
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
