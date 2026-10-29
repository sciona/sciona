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
from sciona.physics_ingest.sound_scale_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1556389363': {'latex_lhs': 'E_{\\\\rm Rydberg}', 'latex_rhs': '\\\\frac{ m_e e^4 }{ 32 \\pi^2 \\epsilon_0^2 \\hbar^2}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0009838')", 'sympy_rhs': "Mul(Rational(1, 32), Pow(Symbol('pdg0001054'), Integer(-2)), Pow(Symbol('pdg0001999'), Integer(4)), Symbol('pdg0002515'), Pow(Symbol('pdg0003141'), Integer(-2)), Pow(Symbol('pdg0007940'), Integer(-2)))"}, '2897612567': {'latex_lhs': 'v', 'latex_rhs': '\\alpha c \\sqrt{ \\\\frac{m_e}{A m_p} }', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Mul(Symbol('pdg0001370'), Symbol('pdg0004567'), Pow(Mul(Symbol('pdg0002515'), Pow(Symbol('pdg0003285'), Integer(-1)), Pow(Symbol('pdg0005916'), Integer(-1))), Rational(1, 2)))"}, '3291685884': {'latex_lhs': 'E', 'latex_rhs': '\\\\frac{ m_e e^4 }{ 32 \\pi^2 \\epsilon_0^2 \\hbar^2}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002241')", 'sympy_rhs': "Mul(Rational(1, 32), Pow(Symbol('pdg0001054'), Integer(-2)), Pow(Symbol('pdg0001999'), Integer(4)), Symbol('pdg0002515'), Pow(Symbol('pdg0003141'), Integer(-2)), Pow(Symbol('pdg0007940'), Integer(-2)))"}, '3935058307': {'latex_lhs': 'v', 'latex_rhs': '\\sqrt{ \\\\frac{m_e}{m} \\\\frac{e^4}{32 \\pi^2 \\epsilon_0^2 \\hbar^2} }', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Mul(Rational(1, 8), Pow(Integer(2), Rational(1, 2)), Pow(Mul(Pow(Symbol('pdg0001054'), Integer(-2)), Pow(Symbol('pdg0001999'), Integer(4)), Symbol('pdg0002515'), Pow(Symbol('pdg0003141'), Integer(-2)), Pow(Symbol('pdg0007940'), Integer(-2)), Pow(Symbol('pdg0009863'), Integer(-1))), Rational(1, 2)))"}, '4107032818': {'latex_lhs': 'E_{\\\\rm Rydberg}', 'latex_rhs': 'E', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0009838')", 'sympy_rhs': "Symbol('pdg0002241')"}, '4560648264': {'latex_lhs': 'v', 'latex_rhs': '\\sqrt{ \\\\frac{K + (4/3) G}{\\\\rho} }', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Pow(Mul(Pow(Symbol('pdg0003935'), Integer(-1)), Add(Symbol('pdg0001466'), Mul(Rational(4, 3), Symbol('pdg0003033')))), Rational(1, 2))"}, '5646314683': {'latex_lhs': 'm', 'latex_rhs': 'A m_p', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0009863')", 'sympy_rhs': "Mul(Symbol('pdg0003285'), Symbol('pdg0005916'))"}, '5789289057': {'latex_lhs': 'v', 'latex_rhs': '\\alpha c \\sqrt{ \\\\frac{m_e}{2 m} }', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 2)), Symbol('pdg0001370'), Symbol('pdg0004567'), Pow(Mul(Symbol('pdg0002515'), Pow(Symbol('pdg0009863'), Integer(-1))), Rational(1, 2)))"}, '5838268428': {'latex_lhs': '\\alpha c', 'latex_rhs': '\\\\frac{1}{4 \\pi \\epsilon_0} \\\\frac{e^2}{\\hbar}', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0001370'), Symbol('pdg0004567'))", 'sympy_rhs': "Mul(Rational(1, 4), Pow(Symbol('pdg0001054'), Integer(-1)), Pow(Symbol('pdg0001999'), Integer(2)), Pow(Symbol('pdg0003141'), Integer(-1)), Pow(Symbol('pdg0007940'), Integer(-1)))"}, '6504442697': {'latex_lhs': 'v', 'latex_rhs': '\\sqrt{ \\\\frac{K}{\\\\rho} }', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Pow(Mul(Symbol('K'), Pow(Symbol('pdg0003935'), Integer(-1))), Rational(1, 2))"}, '7701249282': {'latex_lhs': 'v_u', 'latex_rhs': '\\alpha c \\sqrt{ \\\\frac{m_e}{m_p} }', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004635')", 'sympy_rhs': "Mul(Symbol('pdg0001370'), Symbol('pdg0004567'), Pow(Mul(Symbol('pdg0002515'), Pow(Symbol('pdg0005916'), Integer(-1))), Rational(1, 2)))"}, '7837519722': {'latex_lhs': 'v', 'latex_rhs': '\\sqrt{f} \\sqrt{\\\\frac{E}{m}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0006235'), Rational(1, 2)), Pow(Mul(Symbol('pdg0002241'), Pow(Symbol('pdg0009863'), Integer(-1))), Rational(1, 2)))"}, '8090924099': {'latex_lhs': 'v', 'latex_rhs': '\\sqrt{ \\left( f\\\\frac{E}{a^3} \\\\right) \\\\frac{1}{\\\\rho} }', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Pow(Mul(Symbol('pdg0002241'), Pow(Symbol('pdg0003935'), Integer(-1)), Pow(Symbol('pdg0005854'), Integer(-3)), Symbol('pdg0006235')), Rational(1, 2))"}, '8106885760': {'latex_lhs': '\\alpha', 'latex_rhs': '\\\\frac{1}{4 \\pi \\epsilon_0} \\\\frac{e^2}{\\hbar c}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001370')", 'sympy_rhs': "Mul(Rational(1, 4), Pow(Symbol('pdg0001054'), Integer(-1)), Pow(Symbol('pdg0001999'), Integer(2)), Pow(Symbol('pdg0003141'), Integer(-1)), Pow(Symbol('pdg0004567'), Integer(-1)), Pow(Symbol('pdg0007940'), Integer(-1)))"}, '8688588981': {'latex_lhs': 'a^3 \\\\rho', 'latex_rhs': 'm', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0003935'), Pow(Symbol('pdg0005854'), Integer(3)))", 'sympy_rhs': "Symbol('pdg0009863')"}, '8908736791': {'latex_lhs': '\\\\rho', 'latex_rhs': '\\\\frac{m}{a^3}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0003935')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0005854'), Integer(-3)), Symbol('pdg0009863'))"}, '9376481176': {'latex_lhs': 'K', 'latex_rhs': 'f \\\\frac{E}{a^3}', 'latex_relation': '=', 'sympy_lhs': "Symbol('K')", 'sympy_rhs': "Mul(Symbol('pdg0002241'), Pow(Symbol('pdg0005854'), Integer(-3)), Symbol('pdg0006235'))"}, '9640720571': {'latex_lhs': 'v', 'latex_rhs': '\\\\frac{e^2}{4 \\pi \\epsilon_0 \\hbar} \\sqrt{\\\\frac{m_e}{2 m}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Mul(Rational(1, 8), Pow(Integer(2), Rational(1, 2)), Pow(Symbol('pdg0001054'), Integer(-1)), Pow(Symbol('pdg0001999'), Integer(2)), Pow(Symbol('pdg0003141'), Integer(-1)), Pow(Symbol('pdg0007940'), Integer(-1)), Pow(Mul(Symbol('pdg0002515'), Pow(Symbol('pdg0009863'), Integer(-1))), Rational(1, 2)))"}, '9854442418': {'latex_lhs': 'v', 'latex_rhs': '\\sqrt{\\\\frac{E}{m}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002077')", 'sympy_rhs': "Pow(Mul(Symbol('pdg0002241'), Pow(Symbol('pdg0009863'), Integer(-1))), Rational(1, 2))"}}
EXPECTED_STEPS = [('111782', {'4560648264', '6504442697'}), ('111556', {'8090924099', '9376481176', '6504442697'}), ('111182', {'8688588981', '8908736791'}), ('111634', {'8090924099', '8688588981', '7837519722'}), ('111782', {'9854442418', '7837519722'}), ('111182', {'5838268428', '8106885760'}), ('111556', {'3291685884', '1556389363', '4107032818'}), ('111556', {'3935058307', '3291685884', '9854442418'}), ('111457', {'3935058307', '9640720571'}), ('111556', {'5838268428', '9640720571', '5789289057'}), ('111556', {'2897612567', '5646314683', '5789289057'}), ('111773', {'2897612567', '7701249282'})]
FEEDS = [["StrictGreaterThan(Symbol('pdg0001466'), Symbol('pdg0003033'))"], [], ["Pow(Symbol('pdg0005854'), Integer(3))"], [], ["Mul(Pow(Symbol('pdg0006235'), Rational(1, 2)), Mul(Integer(2), Symbol('approx')))"], ["Symbol('pdg0004567')"], [], [], [], [], [], ["Symbol('pdg0003285')"]]
MISSING = set()


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=12 or len(bindings)!=30 or any(b['status']!='active' for b in bindings):
            raise ValueError('Source inventory differs')
        for i,(node,(rule,identities)) in enumerate(zip(nodes,EXPECTED_STEPS)):
            sig=json.loads(node['type_signature'])
            actual={b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            feeds=[f['sympy'] for f in sig['variable_bindings']['feeds']]
            expected_feeds=FEEDS[i]
            if sig['inference_rule_id']!=rule or actual!=identities or feeds!=expected_feeds:
                raise ValueError('Source per-step binding/rule/feed differs')
        for binding in bindings:
            rows=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s',(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
            identity=binding['bound_artifact_fqdn'].rsplit('.',1)[1]
            if not rows and identity in MISSING: continue
            if len(rows)!=1 or identity in MISSING: raise ValueError('Expected stored coverage differs')
            row=rows[0]; identity=row['source_payload']['id']; raw=row['source_payload']['raw_payload']
            stored_expected = {k:(v.replace(chr(92)*2, chr(92)) if k.startswith('latex_') else v) for k,v in EXPECTED[identity].items()}
            if {k:raw.get(k) for k in stored_expected} != stored_expected:
                raise ValueError('Reviewed stored source fields differ: '+identity)
            evidence=prepare_pdg_evidence(row,symbol_file.read_bytes())
            records[identity]=dict(source_payload_sha256=_digest(row['source_payload']),fresh_source_evidence_sha256=_digest(evidence),equation=EXPECTED[identity])
            pins.append(row['snapshot_payload']['core_file_sha256'])
    if set(records)!=set(EXPECTED)-MISSING or any(p!=pins[0] for p in pins):
        raise ValueError('Reviewed source equation coverage differs')
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
    result=dict(approved=False,source_nodes=12,source_bindings=30,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Restore factor2 inside mass denominator in final two source equations.', 'Bulk-only step requires G/K small, not merely K>G.', 'Dropping sqrt(f) means unit-prefactor scaling, not equality at sqrt(f) approximately2.', 'Rydberg substitution is a bonding-energy scale model.', 'Maximum at A=1 requires A>=1.', 'Reconstruct mathematical pi and unify bulk-modulus aliases.', 'No universal material-speed bound proven.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_sound_scale_source.py','sciona/physics_ingest/sound_scale_proof.py']})


    historical=json.loads((root/'docs/reviews/sound_scale_execution.json').read_text())['source_proof']
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
