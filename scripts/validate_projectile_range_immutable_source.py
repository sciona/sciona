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
from sciona.physics_ingest.projectile_range_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1087417579': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Add(Mul(Integer(-1), Rational(1, 2), Symbol('pdg0001649'), Pow(Symbol('pdg0002467'), Integer(2))), Mul(Symbol('pdg0002467'), Symbol('pdg0005153'), sin(Symbol('pdg0001575'))))", 'latex_lhs': '0', 'latex_rhs': '- \\\\frac{1}{2} g t_f^2 + v_0 t_f \\sin(\\\\theta)', 'latex_relation': '='}, '1191796961': {'sympy_lhs': "Mul(Rational(1, 2), Symbol('pdg0001649'), Symbol('pdg0002467'))", 'sympy_rhs': "Mul(Symbol('pdg0005153'), sin(Symbol('pdg0001575')))", 'latex_lhs': '\\\\frac{1}{2} g t_f', 'latex_rhs': 'v_0 \\sin(\\\\theta)', 'latex_relation': '='}, '1541916015': {'sympy_lhs': "Symbol('pdg0001575')", 'sympy_rhs': "Mul(Rational(1, 4), Symbol('pdg0003141'))", 'latex_lhs': '\\\\theta', 'latex_rhs': '\\\\frac{\\pi}{4}', 'latex_relation': '='}, '1650441634': {'sympy_lhs': "Symbol('pdg0001469')", 'sympy_rhs': 'Integer(0)', 'latex_lhs': 'y_0', 'latex_rhs': '0', 'latex_relation': '='}, '2086924031': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Add(Mul(Integer(-1), Rational(1, 2), Symbol('pdg0001649'), Symbol('pdg0002467')), Mul(Symbol('pdg0005153'), sin(Symbol('pdg0001575'))))", 'latex_lhs': '0', 'latex_rhs': '- \\\\frac{1}{2} g t_f + v_0 \\sin(\\\\theta)', 'latex_relation': '='}, '2297105551': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Integer(2), Pow(Symbol('pdg0001649'), Integer(-1)), Pow(Symbol('pdg0005153'), Integer(2)), sin(Symbol('pdg0001575')), cos(Symbol('pdg0001575')))", 'latex_lhs': 'd', 'latex_rhs': 'v_0 \\\\frac{2 v_0 \\sin(\\\\theta)}{g} \\cos(\\\\theta)', 'latex_relation': '='}, '2378095808': {'sympy_lhs': "Symbol('pdg0003652')", 'sympy_rhs': "Add(Symbol('pdg0001572'), Symbol('pdg0001943'))", 'latex_lhs': 'x_f', 'latex_rhs': 'x_0 + d', 'latex_relation': '='}, '2405307372': {'sympy_lhs': "sin(Mul(Integer(2), Symbol('pdg0001464')))", 'sympy_rhs': "Mul(Integer(2), sin(Symbol('pdg0001464')), cos(Symbol('pdg0001464')))", 'latex_lhs': '\\sin(2 x)', 'latex_rhs': '2 \\sin(x) \\cos(x)', 'latex_relation': '='}, '2519058903': {'sympy_lhs': "sin(Mul(Integer(2), Symbol('pdg0001575')))", 'sympy_rhs': "Mul(Integer(2), sin(Symbol('pdg0001575')), cos(Symbol('pdg0001575')))", 'latex_lhs': '\\sin(2 \\\\theta)', 'latex_rhs': '2 \\sin(\\\\theta) \\cos(\\\\theta)', 'latex_relation': '='}, '3485125659': {'sympy_lhs': "Symbol('pdg0003652')", 'sympy_rhs': "Add(Symbol('pdg0001572'), Mul(Symbol('pdg0002467'), Symbol('pdg0005153'), cos(Symbol('pdg0001575'))))", 'latex_lhs': 'x_f', 'latex_rhs': 'v_0 t_f \\cos(\\\\theta) + x_0', 'latex_relation': '='}, '3607070319': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001649'), Integer(-1)), Pow(Symbol('pdg0005153'), Integer(2)), sin(Mul(Rational(1, 2), Symbol('pdg0003141'))))", 'latex_lhs': 'd', 'latex_rhs': '\\\\frac{v_0^2}{g} \\sin\\left(2 \\\\frac{\\pi}{4}\\\\right)', 'latex_relation': '='}, '4268085801': {'sympy_lhs': "Add(Symbol('pdg0001572'), Symbol('pdg0001943'))", 'sympy_rhs': "Add(Symbol('pdg0001572'), Mul(Symbol('pdg0002467'), Symbol('pdg0005153'), cos(Symbol('pdg0001575'))))", 'latex_lhs': 'x_0 + d', 'latex_rhs': 'v_0 t_f \\cos(\\\\theta) + x_0', 'latex_relation': '='}, '4370074654': {'sympy_lhs': "Symbol('pdg0001467')", 'sympy_rhs': "Symbol('pdg0002467')", 'latex_lhs': 't', 'latex_rhs': 't_f', 'latex_relation': '='}, '4778077984': {'sympy_lhs': "Symbol('pdg0002467')", 'sympy_rhs': "Mul(Integer(2), Pow(Symbol('pdg0001649'), Integer(-1)), Symbol('pdg0005153'), sin(Symbol('pdg0001575')))", 'latex_lhs': 't_f', 'latex_rhs': '\\\\frac{2 v_0 \\sin(\\\\theta)}{g}', 'latex_relation': '='}, '5353282496': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001649'), Integer(-1)), Pow(Symbol('pdg0005153'), Integer(2)))", 'latex_lhs': 'd', 'latex_rhs': '\\\\frac{v_0^2}{g}', 'latex_relation': '='}, '5373931751': {'sympy_lhs': "Symbol('pdg0001467')", 'sympy_rhs': "Symbol('pdg0002467')", 'latex_lhs': 't', 'latex_rhs': 't_f', 'latex_relation': '='}, '5379546684': {'sympy_lhs': "Symbol('pdg0007092')", 'sympy_rhs': "Add(Symbol('pdg0001469'), Mul(Integer(-1), Rational(1, 2), Symbol('pdg0001649'), Pow(Symbol('pdg0002467'), Integer(2))), Mul(Symbol('pdg0002467'), Symbol('pdg0005153'), sin(Symbol('pdg0001575'))))", 'latex_lhs': 'y_f', 'latex_rhs': '- \\\\frac{1}{2} g t_f^2 + v_0 t_f \\sin(\\\\theta) + y_0', 'latex_relation': '='}, '5438722682': {'sympy_lhs': "Symbol('pdg0004037')", 'sympy_rhs': "Add(Mul(Symbol('pdg0001467'), Symbol('pdg0005153'), cos(Symbol('pdg0001575'))), Symbol('pdg0001572'))", 'latex_lhs': 'x', 'latex_rhs': 'v_0 t \\cos(\\\\theta) + x_0', 'latex_relation': '='}, '7233558441': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Symbol('pdg0002467'), Symbol('pdg0005153'), cos(Symbol('pdg0001575')))", 'latex_lhs': 'd', 'latex_rhs': 'v_0 t_f \\cos(\\\\theta)', 'latex_relation': '='}, '8198310977': {'sympy_lhs': 'Integer(0)', 'sympy_rhs': "Add(Symbol('pdg0001469'), Mul(Integer(-1), Rational(1, 2), Symbol('pdg0001649'), Pow(Symbol('pdg0002467'), Integer(2))), Mul(Symbol('pdg0002467'), Symbol('pdg0005153'), sin(Symbol('pdg0001575'))))", 'latex_lhs': '0', 'latex_rhs': '- \\\\frac{1}{2} g t_f^2 + v_0 t_f \\sin(\\\\theta) + y_0', 'latex_relation': '='}, '8922441655': {'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001649'), Integer(-1)), Pow(Symbol('pdg0005153'), Integer(2)), sin(Mul(Integer(2), Symbol('pdg0001575'))))", 'latex_lhs': 'd', 'latex_rhs': '\\\\frac{v_0^2}{g} \\sin(2 \\\\theta)', 'latex_relation': '='}, '9112191201': {'sympy_lhs': "Symbol('pdg0007092')", 'sympy_rhs': 'Integer(0)', 'latex_lhs': 'y_f', 'latex_rhs': '0', 'latex_relation': '='}, '9862900242': {'sympy_lhs': "Symbol('pdg0005647')", 'sympy_rhs': "Add(Mul(Integer(-1), Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(2)), Symbol('pdg0001649')), Mul(Symbol('pdg0001467'), Symbol('pdg0005153'), sin(Symbol('pdg0001575'))), Symbol('pdg0001469'))", 'latex_lhs': 'y', 'latex_rhs': '- \\\\frac{1}{2} g t^2 + v_0 t \\sin(\\\\theta) + y_0', 'latex_relation': '='}}
EXPECTED_STEPS = [('111984', {'9862900242', '5379546684'}), ('111278', {'9112191201', '5373931751'}), ('111355', {'9112191201', '8198310977', '5379546684'}), ('111556', {'8198310977', '1650441634', '1087417579'}), ('111975', {'2086924031', '1087417579'}), ('111530', {'2086924031', '1191796961'}), ('111182', {'4778077984', '1191796961'}), ('111984', {'3485125659', '5438722682'}), ('111278', {'2378095808', '4370074654'}), ('111556', {'2378095808', '4268085801', '3485125659'}), ('111282', {'7233558441', '4268085801'}), ('111556', {'7233558441', '2297105551', '4778077984'}), ('111886', {'2405307372', '2519058903'}), ('111556', {'2297105551', '8922441655', '2519058903'}), ('111773', {'1541916015', '8922441655'}), ('111556', {'3607070319', '1541916015', '8922441655'}), ('111457', {'3607070319', '5353282496'})]
FEEDS = [["Symbol('pdg0005647')", "Symbol('pdg0007092')", "Symbol('pdg0001467')", "Symbol('pdg0002467')"], [], [], [], ["Symbol('pdg0002467')"], ["Mul(Pow(Integer(2), Integer(-1)), Mul(Symbol('pdg0001649'), Symbol('pdg0002467')))"], ["Mul(Integer(2), Pow(Symbol('pdg0001649'), Integer(-1)))"], ["Symbol('pdg0004037')", "Symbol('pdg0003652')", "Symbol('pdg0001467')", "Symbol('pdg0002467')"], [], [], ["Symbol('pdg0001572')"], [], ["Symbol('pdg0001575')", "Symbol('pdg0001464')"], [], ["Symbol('pdg0001575')"], [], []]
MISSING = {'9862900242', '5438722682'}


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash']!=SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=17 or len(bindings)!=40 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=17,source_bindings=40,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Step13 feed order reverses the actual x-to-theta substitution; use the intended angle.', 'Steps2and9 need explicit landing boundary conditions, not consequences of t=tf alone.', 'Step5 divides by flight time: require positive launch speed and 0<theta<pi/2; endpoint ranges are continuous extensions.', 'Use mathematical pi rather than an unconstrained named symbol.', 'Step15 requires fixed speed, positive constant gravity, level ground, no drag and a global maximum argument.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_projectile_range_source.py','sciona/physics_ingest/projectile_range_proof.py']})

    historical=json.loads((root/'docs/reviews/projectile_range_execution.json').read_text())['source_proof']
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
