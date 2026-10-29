"""Fresh immutable source audit; explicit reconstruction, not literal AST parity."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.eigenstate_orthogonality_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1010393944': {'latex_lhs': 'x', 'latex_rhs': '\\langle\\psi_{\\alpha}| a_{\\\\beta} |\\psi_{\\\\beta} \\\\rangle', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001464')", 'sympy_rhs': "Mul(Symbol('pdg0007752'), Function('Bra')(Symbol('pdg0004679')), Function('Ket')(Symbol('pdg0002090')))"}, '1203938249': {'latex_lhs': 'a_{\\\\beta} \\langle \\psi_{\\alpha} | \\psi_{\\\\beta} \\\\rangle', 'latex_rhs': 'a_{\\alpha} \\langle \\psi_{\\alpha} | \\psi_{\\\\beta} \\\\rangle', 'latex_relation': '=', 'sympy_lhs': "Equality(Symbol('pdg0007752')*Bra('pdg0004679')*Ket('pdg0002090'),Symbol('pdg0007752')*Bra('pdg0004679')*Ket('pdg0002090'))", 'sympy_rhs': ''}, '1395858355': {'latex_lhs': 'x', 'latex_rhs': '\\langle \\psi_{\\alpha}| a_{\\alpha} |\\psi_{\\\\beta}\\\\rangle', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001464')", 'sympy_rhs': "Mul(Symbol('pdg0002427'), Function('Bra')(Symbol('pdg0004679')), Function('Ket')(Symbol('pdg0002090')))"}, '2394240499': {'latex_lhs': 'x', 'latex_rhs': 'a_{\\\\beta} \\langle \\psi_{\\alpha} | \\psi_{\\\\beta} \\\\rangle', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001464')", 'sympy_rhs': "Mul(Symbol('pdg0007752'), Function('Bra')(Symbol('pdg0004679')), Function('Ket')(Symbol('pdg0002090')))"}, '2394935831': {'latex_lhs': '( a_{\\\\beta} - a_{\\alpha} ) \\langle \\psi_{\\alpha} | \\psi_{\\\\beta} \\\\rangle', 'latex_rhs': '0', 'latex_relation': '=', 'sympy_lhs': "Mul(Add(Mul(Integer(-1), Symbol('pdg0002427')), Symbol('pdg0007752')), Function('Bra')(Symbol('pdg0004679')), Function('Ket')(Symbol('pdg0002090')))", 'sympy_rhs': 'Integer(0)'}, '3924948349': {'latex_lhs': 'a_{\\\\beta} \\langle \\psi_{\\alpha} | \\psi_{\\\\beta} \\\\rangle - a_{\\alpha} \\langle \\psi_{\\alpha} | \\psi_{\\\\beta} \\\\rangle', 'latex_rhs': '0', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0007752')", 'sympy_rhs': ''}, '3943939590': {'latex_lhs': 'x', 'latex_rhs': 'a_{\\alpha} \\langle \\psi_{\\alpha}| \\psi_{\\\\beta}\\\\rangle', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002427')", 'sympy_rhs': ''}, '9596004948': {'latex_lhs': 'x', 'latex_rhs': '\\langle\\psi_{\\alpha}| \\hat{A} |\\psi_{\\\\beta}\\\\rangle', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001464')", 'sympy_rhs': "Mul(Symbol('pdg0005598'), Function('Bra')(Symbol('pdg0004679')), Function('Ket')(Symbol('pdg0002090')))"}}
EXPECTED_STEPS = [
 ('111946', {'9596004948','1010393944'}),
 ('111390', {'9596004948','1395858355'}),
 ('111457', {'1010393944','2394240499'}),
 ('111457', {'1395858355','3943939590'}),
 ('111355', {'2394240499','3943939590','1203938249'}),
 ('111282', {'1203938249','3924948349'}),
 ('111728', {'3924948349','2394935831'}),
]


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=7 or len(bindings)!=15 or any(b['status']!='active' for b in bindings):
            raise ValueError('Source inventory differs')
        for i,(node,(rule,identities)) in enumerate(zip(nodes,EXPECTED_STEPS)):
            sig=json.loads(node['type_signature'])
            actual={b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            feeds=[f['sympy'] for f in sig['variable_bindings']['feeds']]
            expected_feeds=["Symbol('pdg0002427')*Bra('pdg0004679')*Ket('pdg0002090')"] if i==5 else []
            if sig['inference_rule_id']!=rule or actual!=identities or feeds!=expected_feeds:
                raise ValueError('Source per-step binding/rule/feed differs')
        for binding in bindings:
            rows=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s',(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
            if len(rows)!=1: raise ValueError('Unique pinned expression snapshot required')
            row=rows[0]; identity=row['source_payload']['id']; raw=row['source_payload']['raw_payload']
            stored_expected = {k:(v.replace(chr(92)*2, chr(92)) if k.startswith('latex_') else v) for k,v in EXPECTED[identity].items()}
            if {k:raw.get(k) for k in stored_expected} != stored_expected:
                raise ValueError('Reviewed stored source fields differ: '+identity)
            evidence=prepare_pdg_evidence(row,symbol_file.read_bytes())
            records[identity]=dict(source_payload_sha256=_digest(row['source_payload']),fresh_source_evidence_sha256=_digest(evidence),equation=EXPECTED[identity])
            pins.append(row['snapshot_payload']['core_file_sha256'])
    if set(records)!=set(EXPECTED) or any(p!=pins[0] for p in pins):
        raise ValueError('Eight source equations with consistent pins required')
    for name,path in [('symbols',symbol_file),('infrules',rule_file),('expr_and_feed',expression_file)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pins[0]['conversion_of_data_formats/'+name+'.cypher']:
            raise ValueError('Public source file pin differs')
    rules=load_pinned_rule_contracts(rule_file.read_bytes(),pins[0]['conversion_of_data_formats/infrules.cypher'])
    if any(rule not in rules for rule,_ in EXPECTED_STEPS): raise ValueError('Missing rule contract')
    found=set()
    for block in re.split(r'(?m)^UNWIND ',expression_file.read_text()):
        match=re.match(r'\[\{id:"(\d+)"',block)
        if not match or match[1] not in EXPECTED: continue
        fields=dict(re.findall(r'(\w+):"((?:\\.|[^"\\])*)"',block))
        if match[1] in found or {k:fields.get(k) for k in EXPECTED[match[1]]}!=EXPECTED[match[1]]:
            raise ValueError('Public equation changed/duplicated')
        found.add(match[1])
    if found!=set(EXPECTED): raise ValueError('Public equation missing')
    result=dict(approved=False,source_nodes=7,source_bindings=15,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        corrections=['Replace scalar/commuting representation of A with an ordered matrix product.',
                     'Reconstruct 3943939590 and 3924948349 from LaTeX; stored symbolic RHS fields are empty.',
                     '1203938249 incorrectly repeats beta eigenvalue on both sides; RHS must use alpha.',
                     'Make Hermiticity, two nonzero eigenstates and real eigenvalues explicit.',
                     'x denotes a matrix element, not position; carry arbitrary common operator units.',
                     'No division by eigenvalue gap in the source conclusion; degenerate overlap may be nonzero.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_eigenstate_orthogonality_source.py','sciona/physics_ingest/eigenstate_orthogonality_proof.py']})

    historical=json.loads((root/'docs/reviews/eigenstate_orthogonality_execution.json').read_text())['source_proof']
    for name,digest in historical['implementation_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Historical source implementation changed')
    if result!=historical:raise ValueError('Immutable source qualification differs')
    result['immutable_validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['approved','source_nodes','source_bindings','corrections']}))
