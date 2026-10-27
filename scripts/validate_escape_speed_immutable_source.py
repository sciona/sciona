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
from sciona.physics_ingest.escape_speed_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1143343287': {'sympy_lhs': "Mul(Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0008656'), Integer(2)))", 'latex_lhs': 'G \\\\frac{m_{\\\\rm Earth}}{r_{\\\\rm Earth}}', 'latex_rhs': '\\\\frac{1}{2} v_{\\\\rm escape}^2', 'latex_relation': '='}, '1330874553': {'sympy_lhs': "Symbol('pdg0008656')", 'sympy_rhs': "Mul(Pow(Integer(2), Rational(1, 2)), Pow(Mul(Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005458'), Symbol('pdg0006277')), Rational(1, 2)))", 'latex_lhs': 'v_{\\\\rm escape}', 'latex_rhs': '\\sqrt{2 G \\\\frac{m_{\\\\rm Earth}}{r_{\\\\rm Earth}}}', 'latex_relation': '='}, '1590774089': {'sympy_lhs': "Symbol('pdg0009398')", 'sympy_rhs': "Mul(Symbol('pdg0004202'), Symbol('pdg0009199'))", 'latex_lhs': 'dW', 'latex_rhs': 'F dx', 'latex_relation': '='}, '1840080113': {'sympy_lhs': "Symbol('pdg0001552')", 'sympy_rhs': 'Integer(0)', 'latex_lhs': 'KE_2', 'latex_rhs': '0', 'latex_relation': '='}, '2042298788': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Add(Mul(Rational(1, 2), Symbol('pdg0005156'), Pow(Symbol('pdg0008656'), Integer(2))), Mul(Integer(-1), Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005156'), Symbol('pdg0005458'), Symbol('pdg0006277')))", 'latex_lhs': '0', 'latex_rhs': '-G \\\\frac{m_{\\\\rm Earth} m}{r_{\\\\rm Earth}} + \\\\frac{1}{2} m v_{\\\\rm escape}^2', 'latex_relation': '='}, '2267521164': {'sympy_lhs': "Symbol('pdg0008849')", 'sympy_rhs': 'Integer(0)', 'latex_lhs': 'PE_2', 'latex_rhs': '0', 'latex_relation': '='}, '2503972039': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Add(Symbol('pdg0005332'), Symbol('pdg0006431'))", 'latex_lhs': '0', 'latex_rhs': 'KE_{\\\\rm escape} + PE_{\\\\rm Earth\\ surface}', 'latex_relation': '='}, '2750380042': {'sympy_lhs': "Symbol('pdg0008656')", 'sympy_rhs': "Mul(Integer(-1), Pow(Integer(2), Rational(1, 2)), Pow(Mul(Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005458'), Symbol('pdg0006277')), Rational(1, 2)))", 'latex_lhs': 'v_{\\\\rm escape}', 'latex_rhs': '-\\sqrt{2 G \\\\frac{m_{\\\\rm Earth}}{r_{\\\\rm Earth}}}', 'latex_relation': '='}, '2977457786': {'sympy_lhs': "Mul(Integer(2), Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'sympy_rhs': "Pow(Symbol('pdg0008656'), Integer(2))", 'latex_lhs': '2 G \\\\frac{m_{\\\\rm Earth}}{r_{\\\\rm Earth}}', 'latex_rhs': 'v_{\\\\rm escape}^2', 'latex_relation': '='}, '3846041519': {'sympy_lhs': "Symbol('pdg0006431')", 'sympy_rhs': "Mul(Integer(-1), Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005156'), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'latex_lhs': 'PE_{\\\\rm Earth\\ surface}', 'latex_rhs': '-G \\\\frac{m_{\\\\rm Earth} m}{r_{\\\\rm Earth}}', 'latex_relation': '='}, '4303372136': {'sympy_lhs': "Symbol('pdg0005579')", 'sympy_rhs': "Add(Symbol('pdg0001955'), Symbol('pdg0004093'))", 'latex_lhs': 'E_1', 'latex_rhs': 'KE_1 + PE_1', 'latex_relation': '='}, '4447113478': {'sympy_lhs': "Integral(Integer(1), Tuple(Symbol('pdg0006789')))", 'sympy_rhs': "Mul(Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'), Integral(Pow(Symbol('pdg0004037'), Integer(-2)), Tuple(Symbol('pdg0004037'), Symbol('pdg0003236'), Symbol('infty'))))", 'latex_lhs': '\\int dW', 'latex_rhs': 'G m_1 m_2 \\int_{ r_{\\\\rm Earth} }^{\\infty} \\\\frac{1}{x^2} dx', 'latex_relation': '='}, '5404822208': {'sympy_lhs': "Symbol('pdg0008656')", 'sympy_rhs': "Mul(Pow(Integer(2), Rational(1, 2)), Pow(Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0005156'), Symbol('pdg0006277')), Rational(1, 2)))", 'latex_lhs': 'v_{\\\\rm escape}', 'latex_rhs': '\\sqrt{2 G \\\\frac{m}{r}}', 'latex_relation': '='}, '5732331610': {'sympy_lhs': "Symbol('pdg0006277')", 'sympy_rhs': '', 'latex_lhs': 'W', 'latex_rhs': 'G m_1 m_2 \\left( \\\\frac{1}{x} \\\\bigg\\\\rvert_{ r_{\\\\rm Earth} }^{\\infty} \\\\right)', 'latex_relation': '='}, '5978756813': {'sympy_lhs': "Symbol('pdg0006789')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005156'), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'latex_lhs': 'W', 'latex_rhs': 'G m_{\\\\rm Earth} m \\left( 0 - \\\\frac{-1}{ r_{\\\\rm Earth}} \\\\right)', 'latex_relation': '='}, '6131764194': {'sympy_lhs': "Symbol('W')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0004037'), Integer(-2)), Symbol('pdg0005156'), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'latex_lhs': 'W', 'latex_rhs': 'G m_{\\\\rm Earth} m \\left( \\\\frac{1}{x^2} \\\\bigg\\\\rvert_{ r_{\\\\rm Earth} }^{\\infty} \\\\right)', 'latex_relation': '='}, '6870322215': {'sympy_lhs': "Symbol('pdg0005332')", 'sympy_rhs': "Mul(Rational(1, 2), Symbol('pdg0005156'), Pow(Symbol('pdg0008656'), Integer(2)))", 'latex_lhs': 'KE_{\\\\rm escape}', 'latex_rhs': '\\\\frac{1}{2} m v_{\\\\rm escape}^2', 'latex_relation': '='}, '6935745841': {'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0004037'), Integer(-2)), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'))", 'latex_lhs': 'F', 'latex_rhs': 'G \\\\frac{m_1 m_2}{x^2}', 'latex_relation': '='}, '7573835180': {'sympy_lhs': "Symbol('pdg0006431')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0006789'))", 'latex_lhs': 'PE_{\\\\rm Earth\\ surface}', 'latex_rhs': '-W', 'latex_relation': '='}, '7749253510': {'sympy_lhs': "Symbol('pdg0006789')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005156'), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'latex_lhs': 'W', 'latex_rhs': 'G \\\\frac{m_{\\\\rm Earth} m }{ r_{\\\\rm Earth}}', 'latex_relation': '='}, '7875206161': {'sympy_lhs': "Symbol('pdg0004550')", 'sympy_rhs': "Add(Symbol('pdg0001352'), Symbol('pdg0008849'))", 'latex_lhs': 'E_2', 'latex_rhs': 'KE_2 + PE_2', 'latex_relation': '='}, '8357234146': {'sympy_lhs': "Symbol('pdg0004929')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0001357'), Integer(2)), Symbol('pdg0005156'))", 'latex_lhs': 'KE', 'latex_rhs': '\\\\frac{1}{2} m v^2', 'latex_relation': '='}, '8558338742': {'sympy_lhs': "Symbol('pdg0004550')", 'sympy_rhs': "Symbol('pdg0005579')", 'latex_lhs': 'E_2', 'latex_rhs': 'E_1', 'latex_relation': '='}, '8604483515': {'sympy_lhs': "Symbol('pdg0009398')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0004037'), Integer(-2)), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'), Symbol('pdg0009199'))", 'latex_lhs': 'dW', 'latex_rhs': 'G \\\\frac{m_1 m_2}{x^2} dx', 'latex_relation': '='}, '8960645192': {'sympy_lhs': "Add(Symbol('pdg0001552'), Symbol('pdg0008849'))", 'sympy_rhs': "Add(Symbol('pdg0001955'), Symbol('pdg0004093'))", 'latex_lhs': 'KE_2 + PE_2', 'latex_rhs': 'KE_1 + PE_1', 'latex_relation': '='}, '9412953728': {'sympy_lhs': "Pow(Symbol('pdg0008656'), Integer(2))", 'sympy_rhs': "Mul(Integer(2), Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'latex_lhs': 'v_{\\\\rm escape}^2', 'latex_rhs': '2 G \\\\frac{m_{\\\\rm Earth}}{r_{\\\\rm Earth}}', 'latex_relation': '='}, '9703482302': {'sympy_lhs': "Mul(Pow(Symbol('pdg0003236'), Integer(-1)), Symbol('pdg0005156'), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'sympy_rhs': "Mul(Rational(1, 2), Symbol('pdg0005156'), Pow(Symbol('pdg0008656'), Integer(2)))", 'latex_lhs': 'G \\\\frac{m_{\\\\rm Earth} m}{r_{\\\\rm Earth}}', 'latex_rhs': '\\\\frac{1}{2} m v_{\\\\rm escape}^2', 'latex_relation': '='}, '9749777192': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Add(Symbol('pdg0001955'), Symbol('pdg0004093'))", 'latex_lhs': '0', 'latex_rhs': 'KE_1 + PE_1', 'latex_relation': '='}}
EXPECTED_STEPS = [('111556', {'6935745841', '1590774089', '8604483515'}), ('111408', {'8604483515', '4447113478'}), ('111662', {'5732331610', '4447113478'}), ('111984', {'5732331610', '6131764194'}), ('111457', {'5978756813', '6131764194'}), ('111457', {'5978756813', '7749253510'}), ('111732', {'8960645192', '4303372136', '8558338742', '7875206161'}), ('111732', {'9749777192', '8960645192', '2267521164', '1840080113'}), ('111984', {'9749777192', '2503972039'}), ('111556', {'3846041519', '7749253510', '7573835180'}), ('111984', {'8357234146', '6870322215'}), ('111732', {'3846041519', '6870322215', '2042298788', '2503972039'}), ('111530', {'9703482302', '2042298788'}), ('111457', {'1143343287', '9703482302'}), ('111182', {'1143343287', '2977457786'}), ('111268', {'9412953728', '2977457786'}), ('111524', {'2750380042', '9412953728', '1330874553'}), ('111984', {'5404822208', '1330874553'})]
FEEDS = [[], [], [], ["Symbol('pdg0005022')", "Symbol('pdg0005458')", "Symbol('pdg0004851')", "Symbol('pdg0005156')"], [], [], [], [], ["Symbol('pdg0001955')", "Symbol('pdg0005332')", "Symbol('pdg0004093')", "Symbol('pdg0006431')"], [], ["Symbol('pdg0004929')", "Symbol('pdg0005332')", "Symbol('pdg0001357')", "Symbol('pdg0008656')"], [], ["Mul(Symbol('pdg0006277'), Mul(Pow(Symbol('pdg0003236'), Integer(-1)), Mul(Symbol('pdg0005156'), Symbol('pdg0005458'))))"], [], ['Integer(2)'], [], [], ["Symbol('pdg0005458')", "Symbol('pdg0005156')", "Symbol('pdg0003236')", "Symbol('pdg0002530')"]]
MISSING = {'6935745841', '1590774089', '5404822208', '8357234146'}


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash']!=SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=18 or len(bindings)!=45 or any(b['status']!='active' for b in bindings):
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
            stored_expected = {k:(v.replace(chr(92)*2, chr(92)) if k.startswith('latex') else v) for k,v in EXPECTED[identity].items()}
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
    result=dict(approved=False,source_nodes=18,source_bindings=45,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Positive force/work are quasistatic outward external work against gravity; gravitational work along escape is negative.', 'Step3 antiderivative is -1/x, not +1/x; step4 must preserve -1/x, not 1/x^2.', 'Restore derivative/integral and boundary evaluation structures lost in source ASTs; use mathematical infinity instead of a named infty symbol.', 'Unify final kinetic-energy aliases: source E2 definition uses pdg0001352 while following expressions use pdg0001552.', 'Step14 cancels positive test mass and therefore needs a nonzero-mass premise.', 'Step17 square roots are alternative signed velocities; positive escape speed is the magnitude.', 'Final generic m denotes central mass, not the cancelled test mass; keep distinct runtime names.', 'Fixed central exterior Newtonian field, negligible test mass, no dissipation, zero terminal speed at infinity; no finite-time arrival or trajectory-clearance guarantee.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_escape_speed_source.py','sciona/physics_ingest/escape_speed_proof.py']})


    historical=json.loads((root/'docs/reviews/escape_speed_execution.json').read_text())['source_proof']
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
