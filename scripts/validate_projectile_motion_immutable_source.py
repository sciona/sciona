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
from sciona.physics_ingest.projectile_motion_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1166310428': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Symbol('pdg0005005')", 'latex_lhs': '0 dt', 'latex_rhs': 'd v_x', 'latex_relation': '='}, '1306360899': {'sympy_lhs': "Symbol('pdg0004037')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001467'), Symbol('pdg0002958')), Symbol('pdg0001572'))", 'latex_lhs': 'x', 'latex_rhs': 'v_{0, x} t + x_0', 'latex_relation': '='}, '1405465835': {'sympy_lhs': "Symbol('pdg0005647')", 'sympy_rhs': "Add(Mul(Integer(-1), Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0001649')), Mul(Symbol('pdg0001467'), Symbol('pdg0009107')), Symbol('pdg0001469'))", 'latex_lhs': 'y', 'latex_rhs': '- \\\\frac{1}{2} g t^2 + v_{0, y} t + y_0', 'latex_relation': '='}, '1676472948': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Add(Mul(Integer(-1), Symbol('pdg0002958')), Symbol('pdg0005505'))", 'latex_lhs': '0', 'latex_rhs': 'v_x - v_{0, x}', 'latex_relation': '='}, '1702349646': {'sympy_lhs': "Mul(Integer(-1), Symbol('dt'), Symbol('pdg0001649'))", 'sympy_rhs': "Symbol('pdg0005674')", 'latex_lhs': '-g dt', 'latex_rhs': 'd v_y', 'latex_relation': '='}, '1819663717': {'sympy_lhs': "Symbol('pdg0007159')", 'sympy_rhs': "Derivative(Symbol('pdg0005505'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': 'a_x', 'latex_rhs': '\\\\frac{d}{dt} v_x', 'latex_relation': '='}, '1963253044': {'sympy_lhs': "Mul(Symbol('pdg0002958'), Symbol('pdg0004711'))", 'sympy_rhs': "Symbol('pdg0009199')", 'latex_lhs': 'v_{0, x} dt', 'latex_rhs': 'dx', 'latex_relation': '='}, '1977955751': {'sympy_lhs': "Mul(Integer(-1), Symbol('pdg0001649'))", 'sympy_rhs': "Derivative(Symbol('pdg0009107'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': '-g', 'latex_rhs': '\\\\frac{d}{dt} v_y', 'latex_relation': '='}, '2366691988': {'sympy_lhs': "Integral(Integer(0), Tuple(Symbol('pdg0001467')))", 'sympy_rhs': "Integral(Integer(1), Tuple(Symbol('pdg0005005')))", 'latex_lhs': '\\int 0 dt', 'latex_rhs': '\\int d v_x', 'latex_relation': '='}, '2461349007': {'sympy_lhs': "Add(Mul(Integer(-1), Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0001649')), Mul(Symbol('pdg0001467'), Symbol('pdg0009431')), Symbol('pdg0001469'))", 'sympy_rhs': "Symbol('pdg0005647')", 'latex_lhs': '- \\\\frac{1}{2} g t^2 + v_{0, y} t + y_0', 'latex_rhs': 'y', 'latex_relation': '='}, '2741489181': {'sympy_lhs': "Symbol('pdg0007055')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0001649'))", 'latex_lhs': 'a_y', 'latex_rhs': '-g', 'latex_relation': '='}, '2858549874': {'sympy_lhs': "Add(Mul(Integer(-1), Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0001649')), Mul(Symbol('pdg0001467'), Symbol('pdg0009431')))", 'sympy_rhs': "Add(Mul(Integer(-1), Symbol('pdg0001469')), Symbol('pdg0005647'))", 'latex_lhs': '- \\\\frac{1}{2} g t^2 + v_{0, y} t', 'latex_rhs': 'y - y_0', 'latex_relation': '='}, '3169580383': {'sympy_lhs': "Symbol('pdg0002423')", 'sympy_rhs': "Derivative(Symbol('pdg0006373'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': '\\vec{a}', 'latex_rhs': '\\\\frac{d\\vec{v}}{dt}', 'latex_relation': '='}, '3676159007': {'sympy_lhs': "Mul(Symbol('pdg0002958'), Integral(Integer(1), Tuple(Symbol('pdg0001467'))))", 'sympy_rhs': "Integral(Integer(1), Tuple(Symbol('pdg0001464')))", 'latex_lhs': 'v_{0, x} \\int dt', 'latex_rhs': '\\int dx', 'latex_relation': '='}, '4158986868': {'sympy_lhs': "Symbol('pdg0001467')", 'sympy_rhs': '', 'latex_lhs': 'a_x \\hat{x} + a_y \\hat{y}', 'latex_rhs': '\\\\frac{d\\vec{v}}{dt}', 'latex_relation': '='}, '5349866551': {'sympy_lhs': "Symbol('pdg0006373')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001700'), Symbol('pdg0009107')), Mul(Symbol('pdg0005505'), Symbol('pdg0008339')))", 'latex_lhs': '\\vec{v}', 'latex_rhs': 'v_x \\hat{x} + v_y \\hat{y}', 'latex_relation': '='}, '5438722682': {'sympy_lhs': "Symbol('pdg0004037')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001467'), Symbol('pdg0005153'), cos(Symbol('pdg0001575'))), Symbol('pdg0001572'))", 'latex_lhs': 'x', 'latex_rhs': 'v_0 t \\cos(\\\\theta) + x_0', 'latex_relation': '='}, '6083821265': {'sympy_lhs': "Mul(Symbol('pdg0005153'), cos(Symbol('pdg0001575')))", 'sympy_rhs': "Symbol('pdg0002958')", 'latex_lhs': 'v_0 \\cos(\\\\theta)', 'latex_rhs': 'v_{0, x}', 'latex_relation': '='}, '6134836751': {'sympy_lhs': "Symbol('pdg0002958')", 'sympy_rhs': "Symbol('pdg0005505')", 'latex_lhs': 'v_{0, x}', 'latex_rhs': 'v_x', 'latex_relation': '='}, '6204539227': {'sympy_lhs': "Add(Mul(Integer(-1), Symbol('pdg0001467'), Symbol('pdg0006277')), Symbol('pdg0009431'))", 'sympy_rhs': "Derivative(Symbol('pdg0005647'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': '-g t + v_{0, y}', 'latex_rhs': '\\\\frac{dy}{dt}', 'latex_relation': '='}, '6572039835': {'sympy_lhs': "Add(Mul(Integer(-1), Symbol('pdg0001467'), Symbol('pdg0001649')), Symbol('pdg0009431'))", 'sympy_rhs': "Symbol('pdg0009107')", 'latex_lhs': '-g t + v_{0, y}', 'latex_rhs': 'v_y', 'latex_relation': '='}, '7252338326': {'sympy_lhs': "Symbol('pdg0009107')", 'sympy_rhs': "Derivative(Symbol('pdg0005647'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': 'v_y', 'latex_rhs': '\\\\frac{dy}{dt}', 'latex_relation': '='}, '7376526845': {'sympy_lhs': "sin(Symbol('pdg0001575'))", 'sympy_rhs': "Mul(Symbol('pdg0005153'), Pow(Symbol('pdg0009431'), Integer(-1)))", 'latex_lhs': '\\sin(\\\\theta)', 'latex_rhs': '\\\\frac{v_{0, y}}{v_0}', 'latex_relation': '='}, '7391837535': {'sympy_lhs': "cos(Symbol('pdg0001575'))", 'sympy_rhs': "Mul(Pow(Symbol('pdg0002958'), Integer(-1)), Symbol('pdg0005153'))", 'latex_lhs': '\\cos(\\\\theta)', 'latex_rhs': '\\\\frac{v_{0, x}}{v_0}', 'latex_relation': '='}, '7455581657': {'sympy_lhs': "Symbol('pdg0002958')", 'sympy_rhs': "Derivative(Symbol('pdg0009199'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': 'v_{0, x}', 'latex_rhs': '\\\\frac{dx}{dt}', 'latex_relation': '='}, '7729413831': {'sympy_lhs': "Add(Mul(Symbol('pdg0001700'), Symbol('pdg0007055')), Mul(Symbol('pdg0007159'), Symbol('pdg0008339')))", 'sympy_rhs': "Derivative(Add(Mul(Symbol('pdg0001700'), Symbol('pdg0009107')), Mul(Symbol('pdg0005505'), Symbol('pdg0008339'))), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': 'a_x \\hat{x} + a_y \\hat{y}', 'latex_rhs': '\\\\frac{d}{dt} \\left(v_x \\hat{x} + v_y \\hat{y} \\\\right)', 'latex_relation': '='}, '8145337879': {'sympy_lhs': "Add(Mul(Integer(-1), Symbol('pdg0001467'), Symbol('pdg0001649'), Symbol('pdg0004711')), Mul(Symbol('pdg0004711'), Symbol('pdg0009431')))", 'sympy_rhs': "Symbol('pdg0005842')", 'latex_lhs': '-g t dt + v_{0, y} dt', 'latex_rhs': 'dy', 'latex_relation': '='}, '8228733125': {'sympy_lhs': "Symbol('pdg0007055')", 'sympy_rhs': "Derivative(Symbol('pdg0009107'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': 'a_y', 'latex_rhs': '\\\\frac{d}{dt} v_y', 'latex_relation': '='}, '8460820419': {'sympy_lhs': "Symbol('pdg0005505')", 'sympy_rhs': "Derivative(Symbol('pdg0009199'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': 'v_x', 'latex_rhs': '\\\\frac{dx}{dt}', 'latex_relation': '='}, '8486706976': {'sympy_lhs': "Add(Mul(Symbol('pdg0001467'), Symbol('pdg0002958')), Symbol('pdg0001572'))", 'sympy_rhs': "Symbol('pdg0004037')", 'latex_lhs': 'v_{0, x} t + x_0', 'latex_rhs': 'x', 'latex_relation': '='}, '8584698994': {'sympy_lhs': "Mul(Integer(-1), Symbol('dt'), Symbol('g'))", 'sympy_rhs': "Symbol('pdg0005674')", 'latex_lhs': '-g \\int dt', 'latex_rhs': '\\int d v_y', 'latex_relation': '='}, '8602512487': {'sympy_lhs': "Symbol('pdg0002423')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001700'), Symbol('pdg0007055')), Mul(Symbol('pdg0007159'), Symbol('pdg0008339')))", 'latex_lhs': '\\vec{a}', 'latex_rhs': 'a_x \\hat{x} + a_y \\hat{y}', 'latex_relation': '='}, '8750379055': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Derivative(Symbol('pdg0005505'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'latex_lhs': '0', 'latex_rhs': '\\\\frac{d}{dt} v_x', 'latex_relation': '='}, '8808860551': {'sympy_lhs': "Add(Mul(Integer(-1), Symbol('pdg0001649'), Integral(Symbol('pdg0001467'), Tuple(Symbol('pdg0001467')))), Mul(Symbol('pdg0009431'), Integral(Integer(1), Tuple(Symbol('pdg0001467')))))", 'sympy_rhs': "Integral(Integer(1), Tuple(Symbol('pdg0005647')))", 'latex_lhs': '-g \\int t dt + v_{0, y} \\int dt', 'latex_rhs': '\\int dy', 'latex_relation': '='}, '8949329361': {'sympy_lhs': "Mul(Symbol('pdg0005153'), sin(Symbol('pdg0001575')))", 'sympy_rhs': "Symbol('pdg0009431')", 'latex_lhs': 'v_0 \\sin(\\\\theta)', 'latex_rhs': 'v_{0, y}', 'latex_relation': '='}, '9341391925': {'sympy_lhs': "Symbol('pdg0006091')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001700'), Symbol('pdg0009431')), Mul(Symbol('pdg0002958'), Symbol('pdg0008339')))", 'latex_lhs': '\\vec{v}_0', 'latex_rhs': 'v_{0, x} \\hat{x} + v_{0, y} \\hat{y}', 'latex_relation': '='}, '9707028061': {'sympy_lhs': "Symbol('pdg0007159')", 'sympy_rhs': 'Integer(0)', 'latex_lhs': 'a_x', 'latex_rhs': '0', 'latex_relation': '='}, '9862900242': {'sympy_lhs': "Symbol('pdg0005647')", 'sympy_rhs': "Add(Mul(Integer(-1), Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0001649')), Mul(Symbol('pdg0001467'), Symbol('pdg0005153'), sin(Symbol('pdg0001575'))), Symbol('pdg0001469'))", 'latex_lhs': 'y', 'latex_rhs': '- \\\\frac{1}{2} g t^2 + v_0 t \\sin(\\\\theta) + y_0', 'latex_relation': '='}, '9882526611': {'sympy_lhs': "Mul(Symbol('pdg0001467'), Symbol('pdg0002958'))", 'sympy_rhs': "Add(Mul(Integer(-1), Symbol('pdg0001572')), Symbol('pdg0004037'))", 'latex_lhs': 'v_{0, x} t', 'latex_rhs': 'x - x_0', 'latex_relation': '='}, '9973952056': {'sympy_lhs': "Mul(Integer(-1), Symbol('pdg0001467'), Symbol('pdg0001649'))", 'sympy_rhs': "Add(Mul(Integer(-1), Symbol('pdg0005153')), Symbol('pdg0009431'))", 'latex_lhs': '-g t', 'latex_rhs': 'v_y - v_{0, y}', 'latex_relation': '='}}
EXPECTED_STEPS = [('111556', {'3169580383', '4158986868', '8602512487'}), ('111556', {'5349866551', '7729413831', '4158986868'}), ('111270', {'7729413831', '1819663717', '8228733125'}), ('111556', {'1819663717', '9707028061', '8750379055'}), ('111556', {'2741489181', '1977955751', '8228733125'}), ('111182', {'1977955751', '1702349646'}), ('111608', {'8584698994', '1702349646'}), ('111457', {'8584698994', '9973952056'}), ('111530', {'9973952056', '6572039835'}), ('111556', {'6204539227', '7252338326', '6572039835'}), ('111182', {'6204539227', '8145337879'}), ('111608', {'8145337879', '8808860551'}), ('111457', {'2858549874', '8808860551'}), ('111530', {'2858549874', '2461349007'}), ('111182', {'1166310428', '8750379055'}), ('111608', {'2366691988', '1166310428'}), ('111457', {'1676472948', '2366691988'}), ('111530', {'1676472948', '6134836751'}), ('111556', {'7455581657', '6134836751', '8460820419'}), ('111182', {'1963253044', '7455581657'}), ('111608', {'1963253044', '3676159007'}), ('111457', {'9882526611', '3676159007'}), ('111530', {'9882526611', '8486706976'}), ('111268', {'1306360899', '8486706976'}), ('111295', {'7391837535', '7376526845', '9341391925'}), ('111182', {'6083821265', '7391837535'}), ('111556', {'1306360899', '6083821265', '5438722682'}), ('111182', {'8949329361', '7376526845'}), ('111268', {'2461349007', '1405465835'}), ('111556', {'9862900242', '8949329361', '1405465835'})]
FEEDS = [[], [], [], [], [], ["Symbol('pdg0004711')"], [], [], ["Symbol('pdg0009431')"], [], ["Symbol('pdg0004711')"], [], [], ["Symbol('pdg0001469')"], ["Symbol('pdg0004711')"], [], [], ["Symbol('pdg0002958')"], [], ["Symbol('pdg0004711')"], [], [], ["Symbol('pdg0001572')"], [], ["Symbol('pdg0001575')"], ["Symbol('pdg0005153')"], [], ["Symbol('pdg0005153')"], [], []]
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
        if len(nodes)!=30 or len(bindings)!=70 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=30,source_bindings=70,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Restore time-dependent velocity/position functions, fixed Cartesian basis and blank vector equation AST.',
                     'Correct instantaneous/initial vertical-velocity aliases and the incorrect scalar-speed substitution in the integrated vertical velocity.',
                     'Correct inverted sine/cosine AST quotients to v0y/v0 and v0x/v0 as specified by the public LaTex.',
                     'Align gravity and spatial-position aliases; differentials dx,dy,dv and dt are not independent position/velocity symbols.',
                     'Replace ambiguous indefinite integrations by definite integrals with initial conditions at t=0.',
                     'Require fixed Cartesian basis and constant downward gravity with no drag; include the zero-gravity inertial limit.',
                     'Trigonometric ratio prerequisites require nonzero initial speed; component products independently extend to zero speed.',
                     'Time-based motion includes vertical and backward launches; no division by horizontal velocity or collision cutoff.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_projectile_motion_source.py','sciona/physics_ingest/projectile_motion_proof.py']})


    historical=json.loads((root/'docs/reviews/projectile_motion_execution.json').read_text())['source_proof']
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
