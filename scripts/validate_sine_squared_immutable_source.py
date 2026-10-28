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
from sciona.physics_ingest.sine_squared_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1248277773': {'latex_lhs': '\\cos(2 x)', 'latex_rhs': '1 - 2 (\\sin(x))^2', 'latex_relation': '=', 'sympy_lhs': "cos(Mul(Integer(2), Symbol('pdg0001464')))", 'sympy_rhs': "Add(Integer(1), Mul(Integer(-1), Integer(2), Pow(sin(Symbol('pdg0001464')), Integer(2))))"}, '3285732911': {'latex_lhs': '(\\cos(x))^2', 'latex_rhs': '1-(\\sin(x))^2', 'latex_relation': '=', 'sympy_lhs': "Pow(cos(Symbol('pdg0001464')), Integer(2))", 'sympy_rhs': "Add(Integer(1), Mul(Integer(-1), Pow(sin(Symbol('pdg0001464')), Integer(2))))"}, '4598294821': {'latex_lhs': '\\exp(2 i x)', 'latex_rhs': '(\\cos(x))^2+2i\\cos(x)\\sin(x)-(\\sin(x))^2', 'latex_relation': '=', 'sympy_lhs': "exp(Mul(Integer(2), Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Add(Mul(Integer(2), Symbol('pdg0004621'), sin(Symbol('pdg0001464')), cos(Symbol('pdg0001464'))), Mul(Integer(-1), Pow(sin(Symbol('pdg0001464')), Integer(2))), Pow(cos(Symbol('pdg0001464')), Integer(2)))"}, '4638429483': {'latex_lhs': '\\exp(2 i x)', 'latex_rhs': '(\\cos(x)+ i \\sin(x))(\\cos(x)+ i \\sin(x))', 'latex_relation': '=', 'sympy_lhs': "exp(Mul(Integer(2), Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Pow(Add(Mul(Symbol('pdg0004621'), sin(Symbol('pdg0001464'))), cos(Symbol('pdg0001464'))), Integer(2))"}, '4827492911': {'latex_lhs': '\\cos(2 x)+(\\sin(x))^2', 'latex_rhs': '1 - (\\sin(x))^2', 'latex_relation': '=', 'sympy_lhs': "Add(Pow(sin(Symbol('pdg0001464')), Integer(2)), cos(Mul(Integer(2), Symbol('pdg0001464'))))", 'sympy_rhs': "Add(Integer(1), Mul(Integer(-1), Pow(sin(Symbol('pdg0001464')), Integer(2))))"}, '4838429483': {'latex_lhs': '\\exp(2 i x)', 'latex_rhs': '\\cos(2 x)+i \\sin(2 x)', 'latex_relation': '=', 'sympy_lhs': "exp(Mul(Integer(2), Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Add(Mul(Symbol('pdg0004621'), sin(Mul(Integer(2), Symbol('pdg0001464')))), cos(Mul(Integer(2), Symbol('pdg0001464'))))"}, '4938429483': {'latex_lhs': '\\exp(i x)', 'latex_rhs': '\\cos(x)+i \\sin(x)', 'latex_relation': '=', 'sympy_lhs': "exp(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Add(Mul(Symbol('pdg0004621'), sin(Symbol('pdg0001464'))), cos(Symbol('pdg0001464')))"}, '5832984291': {'latex_lhs': '(\\sin(x))^2 + (\\cos(x))^2', 'latex_rhs': '1', 'latex_relation': '=', 'sympy_lhs': "Add(Pow(sin(Symbol('pdg0001464')), Integer(2)), Pow(cos(Symbol('pdg0001464')), Integer(2)))", 'sympy_rhs': 'Integer(1)'}, '7572664728': {'latex_lhs': '\\cos(2 x) + 2 (\\sin(x))^2', 'latex_rhs': '1', 'latex_relation': '=', 'sympy_lhs': "Add(Mul(Integer(2), Pow(sin(Symbol('pdg0004037')), Integer(2))), cos(Mul(Integer(2), Symbol('pdg0004037'))))", 'sympy_rhs': 'Integer(1)'}, '9482438243': {'latex_lhs': '(\\cos(x))^2', 'latex_rhs': '\\cos(2 x) + (\\sin(x))^2', 'latex_relation': '=', 'sympy_lhs': "Pow(cos(Symbol('pdg0001464')), Integer(2))", 'sympy_rhs': "Add(Pow(sin(Symbol('pdg0001464')), Integer(2)), cos(Mul(Integer(2), Symbol('pdg0001464'))))"}, '9482928242': {'latex_lhs': '\\cos(2 x)', 'latex_rhs': '(\\cos(x))^2 - (\\sin(x))^2', 'latex_relation': '=', 'sympy_lhs': "cos(Mul(Integer(2), Symbol('pdg0001464')))", 'sympy_rhs': "Add(Mul(Integer(-1), Pow(sin(Symbol('pdg0001464')), Integer(2))), Pow(cos(Symbol('pdg0001464')), Integer(2)))"}, '9482928243': {'latex_lhs': '\\cos(2 x) + (\\sin(x))^2', 'latex_rhs': '(\\cos(x))^2', 'latex_relation': '=', 'sympy_lhs': "Add(Pow(sin(Symbol('pdg0001464')), Integer(2)), cos(Mul(Integer(2), Symbol('pdg0001464'))))", 'sympy_rhs': "Pow(cos(Symbol('pdg0001464')), Integer(2))"}, '9483928192': {'latex_lhs': '\\cos(2 x) + i\\sin(2 x)', 'latex_rhs': '(\\cos(x))^2 + 2 i \\cos(x) \\sin(x) - (\\sin(x))^2', 'latex_relation': '=', 'sympy_lhs': "Add(Mul(Symbol('pdg0004621'), sin(Mul(Integer(2), Symbol('pdg0001464')))), cos(Mul(Integer(2), Symbol('pdg0001464'))))", 'sympy_rhs': "Add(Mul(Integer(2), Symbol('pdg0004621'), sin(Symbol('pdg0001464')), cos(Symbol('pdg0001464'))), Mul(Integer(-1), Pow(sin(Symbol('pdg0001464')), Integer(2))), Pow(cos(Symbol('pdg0001464')), Integer(2)))"}, '9889984281': {'latex_lhs': '2 (\\sin(x))^2', 'latex_rhs': '1 - \\cos(2 x)', 'latex_relation': '=', 'sympy_lhs': "Mul(Integer(2), Pow(sin(Symbol('pdg0001464')), Integer(2)))", 'sympy_rhs': "Add(Integer(1), Mul(Integer(-1), cos(Mul(Integer(2), Symbol('pdg0001464')))))"}, '9988949211': {'latex_lhs': '(\\sin(x))^2', 'latex_rhs': '\\\\frac{1 - \\cos(2 x)}{2}', 'latex_relation': '=', 'sympy_lhs': "Pow(sin(Symbol('pdg0001464')), Integer(2))", 'sympy_rhs': "Add(Rational(1, 2), Mul(Integer(-1), Rational(1, 2), cos(Mul(Integer(2), Symbol('pdg0001464')))))"}}
EXPECTED_STEPS = [('111886', {'4838429483', '4938429483'}), ('111355', {'4838429483', '4598294821', '9483928192'}), ('111253', {'4938429483', '4638429483'}), ('111546', {'4598294821', '4638429483'}), ('111198', {'9483928192', '9482928242'}), ('111530', {'9482928242', '9482928243'}), ('111282', {'5832984291', '3285732911'}), ('111268', {'9482438243', '9482928243'}), ('111355', {'9482438243', '4827492911', '3285732911'}), ('111282', {'1248277773', '4827492911'}), ('111530', {'1248277773', '7572664728'}), ('111282', {'7572664728', '9889984281'}), ('111975', {'9988949211', '9889984281'})]
FEEDS = [["Mul(Integer(2), Symbol('pdg0001464'))", "Symbol('pdg0001464')"], [], [], [], [], ["Pow(sin(Symbol('pdg0001464')), Integer(2))"], ["Pow(sin(Symbol('pdg0001464')), Integer(2))"], [], [], ["Pow(sin(Symbol('pdg0001464')), Integer(2))"], ["Mul(Integer(2), Pow(sin(Symbol('pdg0001464')), Integer(2)))"], ["cos(Mul(Integer(2), Symbol('pdg0001464')))"], ['Integer(2)']]
MISSING = {'9988949211', '4938429483'}


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash']!=SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=13 or len(bindings)!=28 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=13,source_bindings=28,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Recover missing source equations from pinned public expressions.', 'Interpret i as imaginary unit and x as finite real dimensionless angle.', 'Step1 feeds reversed: substitute generic x with2x.', 'Equation7572664728 uses position symbolpdg0004037 instead of anglepdg0001464.', 'Execute step3and4 before step2, which consumes their expansion.', 'Euler self-product uses the same premise twice; retain checked Euler and Pythagorean prerequisites.', 'Sine squared does not determine sine sign; avoid a principal-square-root claim.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_sine_squared_source.py','sciona/physics_ingest/sine_squared_proof.py','sciona/physics_ingest/euler_formula_proof.py']})

    historical=json.loads((root/'docs/reviews/sine_squared_execution.json').read_text())['source_proof']
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
