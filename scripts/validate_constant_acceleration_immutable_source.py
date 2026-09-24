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
from sciona.physics_ingest.constant_acceleration_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1259826355': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Add(Mul(Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0009140')), Mul(Symbol('pdg0001467'), Add(Symbol('pdg0001357'), Mul(Integer(-1), Symbol('pdg0001467'), Symbol('pdg0009140')))))", 'latex_lhs': 'd', 'latex_rhs': '(v - a t) t + \\\\frac{1}{2} a t^2', 'latex_relation': '='}, '1265150401': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Symbol('pdg0001467'), Add(Mul(Rational(1, 2), Symbol('pdg0001467'), Symbol('pdg0009140')), Symbol('pdg0005153')))", 'latex_lhs': 'd', 'latex_rhs': '\\\\frac{2 v_0 + a t}{2} t', 'latex_relation': '='}, '1967582749': {'sympy_lhs': "Symbol('pdg0001467')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0009140'), Integer(-1)), Add(Symbol('pdg0001357'), Mul(Integer(-1), Symbol('pdg0005153'))))", 'latex_lhs': 't', 'latex_rhs': '\\\\frac{v - v_0}{a}', 'latex_relation': '='}, '3366703541': {'sympy_lhs': "Symbol('pdg0009140')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0001357'), Mul(Integer(-1), Symbol('pdg0005153'))))", 'latex_lhs': 'a', 'latex_rhs': '\\\\frac{v - v_0}{t}', 'latex_relation': '='}, '3411994811': {'sympy_lhs': "Symbol('pdg0006709')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0001943'))", 'latex_lhs': 'v_{\\\\rm average}', 'latex_rhs': '\\\\frac{d}{t}', 'latex_relation': '='}, '3462972452': {'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001467'), Symbol('pdg0009140')), Symbol('pdg0005153'))", 'latex_lhs': 'v', 'latex_rhs': 'v_0 + a t', 'latex_relation': '='}, '4580545876': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001357'), Symbol('pdg0001467')), Mul(Integer(-1), Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0009140')))", 'latex_lhs': 'd', 'latex_rhs': 'v t - a t^2 + \\\\frac{1}{2} a t^2', 'latex_relation': '='}, '4748157455': {'sympy_lhs': "Mul(Symbol('pdg0001467'), Symbol('pdg0009140'))", 'sympy_rhs': "Add(Symbol('pdg0001357'), Mul(Integer(-1), Symbol('pdg0005153')))", 'latex_lhs': 'a t', 'latex_rhs': 'v - v_0', 'latex_relation': '='}, '4798787814': {'sympy_lhs': "Add(Mul(Symbol('pdg0001467'), Symbol('pdg0009140')), Symbol('pdg0005153'))", 'sympy_rhs': "Symbol('pdg0001357')", 'latex_lhs': 'a t + v_0', 'latex_rhs': 'v', 'latex_relation': '='}, '4948763856': {'sympy_lhs': "Add(Mul(Integer(2), Symbol('pdg0001943'), Symbol('pdg0009140')), Pow(Symbol('pdg0005153'), Integer(2)))", 'sympy_rhs': "Pow(Symbol('pdg0001357'), Integer(2))", 'latex_lhs': '2 a d + v_0^2', 'latex_rhs': 'v^2', 'latex_relation': '='}, '5144263777': {'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': '', 'latex_lhs': 'v^2', 'latex_rhs': 'v_0^2 + 2 a \\left( v_0 t +\\\\frac{1}{2} a t^2 \\\\right)', 'latex_relation': '='}, '5611024898': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0009140'), Integer(-1)), Add(Pow(Symbol('pdg0001357'), Integer(2)), Mul(Integer(-1), Pow(Symbol('pdg0005153'), Integer(2)))))", 'latex_lhs': 'd', 'latex_rhs': '\\\\frac{1}{2 a} (v^2 - v_0^2)', 'latex_relation': '='}, '5733721198': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0009140'), Integer(-1)), Add(Symbol('pdg0001357'), Mul(Integer(-1), Symbol('pdg0005153'))), Add(Symbol('pdg0001357'), Symbol('pdg0005153')))", 'latex_lhs': 'd', 'latex_rhs': '\\\\frac{1}{2} (v + v_0) \\left( \\\\frac{v - v_0}{a} \\\\right)', 'latex_relation': '='}, '6175547907': {'sympy_lhs': "Symbol('pdg0006709')", 'sympy_rhs': "Add(Mul(Rational(1, 2), Symbol('pdg0001357')), Mul(Rational(1, 2), Symbol('pdg0005153')))", 'latex_lhs': 'v_{\\\\rm average}', 'latex_rhs': '\\\\frac{v + v_0}{2}', 'latex_relation': '='}, '6421241247': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001357'), Symbol('pdg0001467')), Mul(Integer(-1), Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0009140')))", 'latex_lhs': 'd', 'latex_rhs': 'v t - \\\\frac{1}{2} a t^2', 'latex_relation': '='}, '6457044853': {'sympy_lhs': "Add(Symbol('pdg0001357'), Mul(Integer(-1), Symbol('pdg0001467'), Symbol('pdg0009140')))", 'sympy_rhs': "Symbol('pdg0005153')", 'latex_lhs': 'v - a t', 'latex_rhs': 'v_0', 'latex_relation': '='}, '7011114072': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Symbol('pdg0001467'), Add(Mul(Rational(1, 2), Symbol('pdg0001467'), Symbol('pdg0009140')), Symbol('pdg0005153')))", 'latex_lhs': 'd', 'latex_rhs': '\\\\frac{(v_0 + a t) + v_0}{2} t', 'latex_relation': '='}, '7215099603': {'sympy_lhs': "Pow(Symbol('pdg0001357'), Integer(2))", 'sympy_rhs': "Add(Mul(Pow(Symbol('pdg0001467'), Integer(2)), Pow(Symbol('pdg0009140'), Integer(2))), Mul(Integer(2), Symbol('pdg0001467'), Symbol('pdg0005153'), Symbol('pdg0009140')), Pow(Symbol('pdg0005153'), Integer(2)))", 'latex_lhs': 'v^2', 'latex_rhs': 'v_0^2 + 2 a t v_0 + a^2 t^2', 'latex_relation': '='}, '7939765107': {'sympy_lhs': "Pow(Symbol('pdg0001357'), Integer(2))", 'sympy_rhs': "Add(Mul(Integer(2), Symbol('pdg0001943'), Symbol('pdg0009140')), Pow(Symbol('pdg0005153'), Integer(2)))", 'latex_lhs': 'v^2', 'latex_rhs': 'v_0^2 + 2 a d', 'latex_relation': '='}, '8269198922': {'sympy_lhs': "Mul(Integer(2), Symbol('pdg0001943'), Symbol('pdg0009140'))", 'sympy_rhs': "Add(Pow(Symbol('pdg0001357'), Integer(2)), Mul(Integer(-1), Pow(Symbol('pdg0005153'), Integer(2))))", 'latex_lhs': '2 a d', 'latex_rhs': 'v^2 - v_0^2', 'latex_relation': '='}, '8706092970': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Symbol('pdg0001467'), Add(Mul(Rational(1, 2), Symbol('pdg0001357')), Mul(Rational(1, 2), Symbol('pdg0005153'))))", 'latex_lhs': 'd', 'latex_rhs': '\\left(\\\\frac{v + v_0}{2}\\\\right)t', 'latex_relation': '='}, '9658195023': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Add(Mul(Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0009140')), Mul(Symbol('pdg0001467'), Symbol('pdg0005153')))", 'latex_lhs': 'd', 'latex_rhs': 'v_0 t + \\\\frac{1}{2} a t^2', 'latex_relation': '='}, '9759901995': {'sympy_lhs': "Add(Symbol('pdg0001357'), Mul(Integer(-1), Symbol('pdg0005153')))", 'sympy_rhs': "Mul(Symbol('pdg0001467'), Symbol('pdg0009140'))", 'latex_lhs': 'v - v_0', 'latex_rhs': 'a t', 'latex_relation': '='}, '9897284307': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0001943'))", 'sympy_rhs': "Add(Mul(Rational(1, 2), Symbol('pdg0001357')), Mul(Rational(1, 2), Symbol('pdg0005153')))", 'latex_lhs': '\\\\frac{d}{t}', 'latex_rhs': '\\\\frac{v + v_0}{2}', 'latex_relation': '='}}
EXPECTED_STEPS = [('111182', {'4748157455', '3366703541'}), ('111530', {'4748157455', '4798787814'}), ('111268', {'3462972452', '4798787814'}), ('111355', {'6175547907', '9897284307', '3411994811'}), ('111182', {'9897284307', '8706092970'}), ('111634', {'7011114072', '3462972452', '8706092970'}), ('111457', {'7011114072', '1265150401'}), ('111457', {'9658195023', '1265150401'}), ('111483', {'7215099603', '3462972452'}), ('111457', {'7215099603', '5144263777'}), ('111634', {'7939765107', '9658195023', '5144263777'}), ('111282', {'9759901995', '3462972452'}), ('111268', {'4748157455', '9759901995'}), ('111975', {'4748157455', '1967582749'}), ('111634', {'5733721198', '8706092970', '1967582749'}), ('111457', {'5611024898', '5733721198'}), ('111182', {'5611024898', '8269198922'}), ('111530', {'4948763856', '8269198922'}), ('111268', {'4948763856', '7939765107'}), ('111282', {'6457044853', '3462972452'}), ('111634', {'6457044853', '9658195023', '1259826355'}), ('111457', {'1259826355', '4580545876'}), ('111457', {'6421241247', '4580545876'})]
FEEDS = [["Symbol('pdg0001467')"], ["Symbol('pdg0005153')"], [], [], ["Symbol('pdg0001467')"], [], [], [], ['Integer(2)'], [], [], ["Symbol('pdg0005153')"], [], ["Symbol('pdg0009140')"], [], [], ["Mul(Integer(2), Symbol('pdg0009140'))"], ["Pow(Symbol('pdg0005153'), Integer(2))"], [], ["Mul(Symbol('pdg0009140'), Symbol('pdg0001467'))"], [], [], []]
MISSING = {'3411994811'}


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.artifact_id::text,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if graph != dict(artifact_id='95f290e0-a920-5569-bc87-ab25e62f6562',content_hash=SOURCE_HASH):
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=23 or len(bindings)!=51 or any(b['status']!='active' for b in bindings):
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
    return dict(approved=False,source_nodes=23,source_bindings=51,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Restore squared-velocity LHS and missing RHS of expression5144263777 from its pinned LaTex.', 'Require constant acceleration: endpoint-average velocity is not generally the time-average velocity.', 'Treat d as signed displacement, not total distance traveled, and velocities as signed one-dimensional components.', 'Initial quotient definitions require t nonzero; their polynomial consequences extend to t=0 by direct integration.', 'Steps14to16 divide by acceleration and require a nonzero; retain a=0 through the independent polynomial solution.', 'Squaring velocity preserves a consequence but loses sign as an inverse relation; do not infer a unique velocity from v^2 alone.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_constant_acceleration_immutable_source.py','sciona/physics_ingest/constant_acceleration_proof.py']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['approved','source_nodes','source_bindings','corrections']}))
