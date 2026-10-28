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
from sciona.physics_ingest.spring_mass_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1772973171': {'latex_lhs': '-\\\\frac{k}{m} x', 'latex_rhs': '-A \\omega^2 \\cos(\\omega t)', 'latex_relation': '=', 'sympy_lhs': "Mul(Integer(-1), Symbol('k'), Pow(Symbol('pdg0005156'), Integer(-1)), Symbol('x'))", 'sympy_rhs': "Mul(Integer(-1), Symbol('A'), Pow(Symbol('pdg0002321'), Integer(2)), cos(Mul(Symbol('pdg0002321'), Symbol('pdg0009491'))))"}, '1784114349': {'latex_lhs': '\\sqrt{\\\\frac{k}{m}}', 'latex_rhs': '\\omega', 'latex_relation': '=', 'sympy_lhs': "Pow(Mul(Symbol('pdg0001356'), Pow(Symbol('pdg0005156'), Integer(-1))), Rational(1, 2))", 'sympy_rhs': "Symbol('pdg0002321')"}, '1888494137': {'latex_lhs': '-\\sqrt{\\\\frac{k}{m}}', 'latex_rhs': '\\omega', 'latex_relation': '=', 'sympy_lhs': "Mul(Integer(-1), Pow(Mul(Symbol('pdg0001356'), Pow(Symbol('pdg0005156'), Integer(-1))), Rational(1, 2)))", 'sympy_rhs': "Symbol('pdg0002321')"}, '1931103031': {'latex_lhs': '\\\\frac{k}{m}', 'latex_rhs': '\\omega^2', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0001356'), Pow(Symbol('pdg0005156'), Integer(-1)))", 'sympy_rhs': "Pow(Symbol('pdg0002321'), Integer(2))"}, '2148049269': {'latex_lhs': '-\\\\frac{k}{m} A \\cos(\\omega t)', 'latex_rhs': '-A \\omega^2 \\cos(\\omega t)', 'latex_relation': '=', 'sympy_lhs': "Mul(Integer(-1), Symbol('A'), Symbol('k'), Pow(Symbol('pdg0005156'), Integer(-1)), cos(Mul(Symbol('pdg0002321'), Symbol('pdg0009491'))))", 'sympy_rhs': "Mul(Integer(-1), Symbol('A'), Pow(Symbol('pdg0002321'), Integer(2)), cos(Mul(Symbol('pdg0002321'), Symbol('pdg0009491'))))"}, '2334518266': {'latex_lhs': 'm a', 'latex_rhs': '-k x', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0005156'), Symbol('pdg0009140'))", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0001356'), Symbol('pdg0004037'))"}, '4428528271': {'latex_lhs': 'F_{\\\\rm{spring}}', 'latex_rhs': '-k x', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004183')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0001356'), Symbol('pdg0004037'))"}, '5345738321': {'latex_lhs': 'F', 'latex_rhs': 'm a', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Symbol('pdg0005156'), Symbol('pdg0009140'))"}, '5415824175': {'latex_lhs': 'x(t)', 'latex_rhs': 'A \\cos(\\omega t)', 'latex_relation': '=', 'sympy_lhs': "Function('x')(Symbol('pdg0001467'))", 'sympy_rhs': "Mul(Symbol('pdg0009885'), cos(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'))))"}, '5945893986': {'latex_lhs': '\\\\frac{d^2 x}{dt^2}', 'latex_rhs': '-A \\omega^2 \\cos(\\omega t)', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('d'), Integer(2)), Pow(Symbol('dt'), Integer(-2)), Symbol('x'))", 'sympy_rhs': "Mul(Integer(-1), Pow(Symbol('pdg0002321'), Integer(2)), Symbol('pdg0009885'), cos(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'))))"}, '6831694380': {'latex_lhs': 'a', 'latex_rhs': '\\\\frac{d^2 x}{dt^2}', 'latex_relation': '=', 'sympy_lhs': "Symbol('a')", 'sympy_rhs': "Mul(Pow(Symbol('d'), Integer(2)), Pow(Symbol('dt'), Integer(-2)), Symbol('x'))"}, '6908055431': {'latex_lhs': 'x(t)', 'latex_rhs': 'A \\cos\\left(\\\\frac{k}{m} t\\\\right)', 'latex_relation': '=', 'sympy_lhs': "Function('x')(Symbol('pdg0001467'))", 'sympy_rhs': "Mul(Symbol('pdg0009885'), cos(Mul(Symbol('k'), Symbol('pdg0001467'), Pow(Symbol('pdg0005156'), Integer(-1)))))"}, '7652131521': {'latex_lhs': '\\\\frac{dx}{dt}', 'latex_rhs': '-A \\omega \\sin (\\omega t)', 'latex_relation': '=', 'sympy_lhs': "Derivative(Symbol('pdg0004037'), Tuple(Symbol('pdg0001467'), Integer(1)))", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0002321'), Symbol('pdg0009885'), sin(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'))))"}, '8655294002': {'latex_lhs': 'a', 'latex_rhs': '-\\\\frac{k}{m}x', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0009140')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0001356'), Symbol('pdg0004037'), Pow(Symbol('pdg0005156'), Integer(-1)))"}, '8991236357': {'latex_lhs': '\\\\frac{d^2 x}{dt^2}', 'latex_rhs': '-\\\\frac{k}{m} x', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('d'), Integer(2)), Pow(Symbol('dt'), Integer(-2)), Symbol('pdg0004037'))", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0001356'), Symbol('pdg0004037'), Pow(Symbol('pdg0005156'), Integer(-1)))"}}
EXPECTED_STEPS = [('111355', {'2334518266', '5345738321', '4428528271'}), ('111975', {'2334518266', '8655294002'}), ('111355', {'8991236357', '6831694380', '8655294002'}), ('111237', {'5415824175', '8991236357'}), ('111649', {'5415824175', '7652131521'}), ('111649', {'5945893986', '7652131521'}), ('111355', {'1772973171', '8991236357', '5945893986'}), ('111556', {'5415824175', '2148049269', '1772973171'}), ('111182', {'1931103031', '2148049269'}), ('111524', {'1888494137', '1784114349', '1931103031'}), ('111634', {'5415824175', '1784114349', '6908055431'})]
FEEDS = [[], ["Symbol('pdg0005156')"], [], [], ["Symbol('t')"], ["Symbol('pdg0001467')"], [], [], ["Symbol('pdg0001467')"], [], []]
MISSING = {'5345738321'}


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=11 or len(bindings)!=28 or any(b['status']!='active' for b in bindings):
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
        raise ValueError('Fourteen stored and one recovered source equations required')
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
    result=dict(approved=False,source_nodes=11,source_bindings=28,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Final source cosine argument k*t/m omits square root; use sqrt(k/m)*t.', 'Net force and spring force use distinct source IDs; equality requires spring to be the net restoring force.', 'Reconstruct true time derivatives and x(t), replacing scalar differential quotients and inconsistent symbol aliases.', 'First derivative feed plain t and second derivative feed pdg0001467 must denote the same physical time.', 'Step9 multiplier feed AST is time, but LaTeX is -1/(A*cos(omega*t)); replace singular pointwise division by nonzero-amplitude identity evaluated at t=0.', 'Square-root outputs are alternative signed branches, not simultaneous constraints; both give identical cosine motion.', 'Cosine ansatz requires zero initial velocity; zero amplitude is verified independently without dividing by A.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_spring_mass_source.py','sciona/physics_ingest/spring_mass_proof.py']})


    historical=json.loads((root/'docs/reviews/spring_mass_execution.json').read_text())['source_proof']
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
