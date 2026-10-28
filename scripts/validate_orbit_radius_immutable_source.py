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
from sciona.physics_ingest.orbit_radius_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1559688463': {'latex_lhs': '\\left(\\\\frac{T_{\\\\rm geostationary\\ orbit}^2 G m_{\\\\rm Earth}}{4 \\pi^2}\\\\right)^{1/3}', 'latex_rhs': 'r_{\\\\rm geostationary\\ orbit}', 'latex_relation': '=', 'sympy_lhs': "Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 3)), Pow(Mul(Pow(Symbol('pdg0003141'), Integer(-2)), Symbol('pdg0005458'), Pow(Symbol('pdg0005595'), Integer(2)), Symbol('pdg0006277')), Rational(1, 3)))", 'sympy_rhs': "Symbol('pdg0007110')"}, '1994296484': {'latex_lhs': 'v_{\\\\rm satellite}^2', 'latex_rhs': 'G \\\\frac{m_{\\\\rm Earth}}{r}', 'latex_relation': '=', 'sympy_lhs': "Pow(Symbol('pdg0004082'), Integer(2))", 'sympy_rhs': "Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0005458'), Symbol('pdg0006277'))"}, '2617541067': {'latex_lhs': '\\left(\\\\frac{T_{\\\\rm orbit}^2 G m_{\\\\rm Earth}}{4 \\pi^2}\\\\right)^{1/3}', 'latex_rhs': 'r', 'latex_relation': '=', 'sympy_lhs': "Mul(Rational(1, 2), Pow(Integer(2), Rational(1, 3)), Pow(Mul(Pow(Symbol('pdg0003141'), Integer(-2)), Symbol('pdg0005458'), Symbol('pdg0006277'), Pow(Symbol('pdg0008762'), Integer(2))), Rational(1, 3)))", 'sympy_rhs': "Symbol('pdg0002530')"}, '3176662571': {'latex_lhs': 'F_{\\\\rm centripetal}', 'latex_rhs': 'F_{\\\\rm gravity}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002867')", 'sympy_rhs': "Symbol('pdg0001687')"}, '3614055652': {'latex_lhs': 'v', 'latex_rhs': '\\\\frac{2 \\pi r}{T_{\\\\rm orbit}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': "Mul(Integer(2), Symbol('pdg0002530'), Symbol('pdg0003141'), Pow(Symbol('pdg0008762'), Integer(-1)))"}, '3906710072': {'latex_lhs': 'G \\\\frac{m_{\\\\rm Earth}}{r}', 'latex_rhs': '\\\\frac{4 \\pi^2 r^2}{T_{\\\\rm orbit}^2}', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'sympy_rhs': "Mul(Integer(4), Pow(Symbol('pdg0002530'), Integer(2)), Pow(Symbol('pdg0003141'), Integer(2)), Pow(Symbol('pdg0008762'), Integer(-2)))"}, '4072200527': {'latex_lhs': '\\\\frac{m_{\\\\rm satellite} v_{\\\\rm satellite}^2}{r}', 'latex_rhs': 'G \\\\frac{m_{\\\\rm Earth} m_{\\\\rm satellite}}{r^2}', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0003569'), Pow(Symbol('pdg0004082'), Integer(2)))", 'sympy_rhs': "Mul(Pow(Symbol('pdg0002530'), Integer(-2)), Symbol('pdg0003569'), Symbol('pdg0005458'), Symbol('pdg0006277'))"}, '4245712581': {'latex_lhs': 'v', 'latex_rhs': '\\\\frac{2 \\pi r}{t}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': "Mul(Integer(2), Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0002530'), Symbol('pdg0003141'))"}, '4627284246': {'latex_lhs': 'F_{\\\\rm centripetal}', 'latex_rhs': '\\\\frac{m_{\\\\rm satellite} v_{\\\\rm satellite}^2}{r}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001687')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0003569'), Pow(Symbol('pdg0004082'), Integer(2)))"}, '4858693811': {'latex_lhs': '\\\\frac{T_{\\\\rm orbit}^2 G m_{\\\\rm Earth}}{4 \\pi^2}', 'latex_rhs': 'r^3', 'latex_relation': '=', 'sympy_lhs': "Mul(Rational(1, 4), Pow(Symbol('pdg0003141'), Integer(-2)), Symbol('pdg0005458'), Symbol('pdg0006277'), Pow(Symbol('pdg0008762'), Integer(2)))", 'sympy_rhs': "Pow(Symbol('pdg0002530'), Integer(3))"}, '5426308937': {'latex_lhs': 'v', 'latex_rhs': '\\\\frac{d}{t}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0001943'))"}, '5563580265': {'latex_lhs': 'F_{\\\\rm gravity}', 'latex_rhs': 'G \\\\frac{m_{\\\\rm Earth} m_{\\\\rm satellite}}{r^2}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0002867')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0002530'), Integer(-2)), Symbol('pdg0003569'), Symbol('pdg0005458'), Symbol('pdg0006277'))"}, '6785303857': {'latex_lhs': 'C', 'latex_rhs': '2 \\pi r', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0003034')", 'sympy_rhs': "Mul(Integer(2), Symbol('pdg0002530'), Symbol('pdg0003141'))"}, '6935745841': {'latex_lhs': 'F', 'latex_rhs': 'G \\\\frac{m_1 m_2}{x^2}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0004037'), Integer(-2)), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'))"}, '7010294143': {'latex_lhs': 'T_{\\\\rm orbit}^2 G m_{\\\\rm Earth}', 'latex_rhs': '4 \\pi^2 r^3', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0005458'), Symbol('pdg0006277'), Pow(Symbol('pdg0008762'), Integer(2)))", 'sympy_rhs': "Mul(Integer(4), Pow(Symbol('pdg0002530'), Integer(3)), Pow(Symbol('pdg0003141'), Integer(2)))"}, '8059639673': {'latex_lhs': 'v^2', 'latex_rhs': '\\\\frac{4 \\pi^2 r^2}{T_{\\\\rm orbit}^2}', 'latex_relation': '=', 'sympy_lhs': "Pow(Symbol('pdg0001357'), Integer(2))", 'sympy_rhs': "Mul(Integer(4), Pow(Symbol('pdg0002530'), Integer(2)), Pow(Symbol('pdg0003141'), Integer(2)), Pow(Symbol('pdg0008762'), Integer(-2)))"}, '9226945488': {'latex_lhs': 'F', 'latex_rhs': '\\\\frac{m v^2}{r}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001357'), Integer(2)), Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0005156'))"}, '9262596735': {'latex_lhs': 'd', 'latex_rhs': '2 \\pi r', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001943')", 'sympy_rhs': "Mul(Integer(2), Symbol('pdg0002530'), Symbol('pdg0003141'))"}}
EXPECTED_STEPS = [('111777', {'6935745841', '5563580265'}), ('111236', {'4627284246', '9226945488'}), ('111732', {'4627284246', '4072200527', '5563580265', '3176662571'}), ('111886', {'9262596735', '6785303857'}), ('111556', {'4245712581', '9262596735', '5426308937'}), ('111886', {'4245712581', '3614055652'}), ('111483', {'3614055652', '8059639673'}), ('111975', {'4072200527', '1994296484'}), ('111355', {'3906710072', '1994296484', '8059639673'}), ('111182', {'3906710072', '7010294143'}), ('111975', {'7010294143', '4858693811'}), ('111483', {'2617541067', '4858693811'}), ('111984', {'1559688463', '2617541067'})]
FEEDS = [["Symbol('pdg0004202')", "Symbol('pdg0002867')", "Symbol('pdg0005458')", "Symbol('pdg0005458')", "Symbol('pdg0004851')", "Symbol('pdg0003569')", "Symbol('pdg0004037')", "Symbol('pdg0002530')"], ["Symbol('pdg0004202')", "Symbol('pdg0001687')", "Symbol('pdg0005156')", "Symbol('pdg0003569')", "Symbol('pdg0001357')", "Symbol('pdg0004082')"], [], ["Symbol('pdg0003034')", "Symbol('pdg0001943')"], [], ["Symbol('pdg0001467')", "Symbol('pdg0008762')"], ['Integer(2)'], ["Mul(Symbol('pdg0003569'), Pow(Symbol('pdg0002530'), Integer(-1)))"], [], ["Mul(Pow(Symbol('pdg0008762'), Integer(2)), Symbol('pdg0002530'))"], ["Mul(Integer(4), Pow(Symbol('pdg0003141'), Integer(2)))"], ['Pow(Integer(3), Integer(-1))'], ["Symbol('pdg0008762')", "Symbol('pdg0005595')", "Symbol('pdg0002530')", "Symbol('pdg0007110')"]]
MISSING = {'6785303857'}


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=13 or len(bindings)!=30 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=13,source_bindings=30,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Recover missing source from pinned public equations.', 'Step1 m1 feed repeats Earth target symbol; rename actual pdg0005022 to pdg0005458.', 'Force-equality AST reverses LaTeX sides; equality is symmetric, choose centripetal=gravity consistently.', 'Unify generic speed v and satellite speed before equal-LHS comparison.', 'Interpret pi as constant; positive-domain cube root yields unique positive radius.', 'Circular test-mass model and nonzero satellite mass required for cancellation.', 'Geostationary label needs equatorial prograde circular orbit and sidereal period; radius is not altitude.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_orbit_radius_source.py','sciona/physics_ingest/orbit_radius_proof.py']})


    historical=json.loads((root/'docs/reviews/orbit_radius_execution.json').read_text())['source_proof']
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
