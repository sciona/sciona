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
from sciona.physics_ingest.euler_formula_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'4923339482': {'latex_lhs': 'i x', 'latex_rhs': '\\log(y)', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0001464'), Symbol('pdg0004621'))", 'sympy_rhs': "log(Symbol('pdg0001452'), Integer(10))"}, '4928239482': {'latex_lhs': '\\log(y)', 'latex_rhs': 'i x', 'latex_relation': '=', 'sympy_lhs': "log(Symbol('pdg0001452'), Integer(10))", 'sympy_rhs': "Mul(Symbol('pdg0001464'), Symbol('pdg0004621'))"}, '4938429483': {'latex_lhs': '\\exp(i x)', 'latex_rhs': '\\cos(x)+i \\sin(x)', 'latex_relation': '=', 'sympy_lhs': "exp(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Add(Mul(Symbol('pdg0004621'), sin(Symbol('pdg0001464'))), cos(Symbol('pdg0001464')))"}, '9429829482': {'latex_lhs': '\\\\frac{d}{dx} y', 'latex_rhs': '-\\sin(x) + i\\cos(x)', 'latex_relation': '=', 'sympy_lhs': "Derivative(Symbol('pdg0001452'), Tuple(Symbol('pdg0001464'), Integer(1)))", 'sympy_rhs': "Add(Mul(Symbol('pdg0004621'), cos(Symbol('pdg0001464'))), Mul(Integer(-1), sin(Symbol('pdg0001464'))))"}, '9482113948': {'latex_lhs': '\\\\frac{dy}{y}', 'latex_rhs': 'i dx', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004621')", 'sympy_rhs': ''}, '9482923849': {'latex_lhs': '\\exp(i x)', 'latex_rhs': 'y', 'latex_relation': '=', 'sympy_lhs': "exp(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Symbol('pdg0001452')"}, '9482943948': {'latex_lhs': '\\log(y)', 'latex_rhs': 'i dx', 'latex_relation': '=', 'sympy_lhs': "log(Symbol('pdg0001452'), Integer(10))", 'sympy_rhs': "Mul(Symbol('pdg0004621'), Symbol('pdg0009199'))"}, '9482984922': {'latex_lhs': '\\\\frac{d}{dx} y', 'latex_rhs': '(i\\sin(x) + \\cos(x)) i', 'latex_relation': '=', 'sympy_lhs': "Derivative(Symbol('pdg0001452'), Tuple(Symbol('pdg0001464'), Integer(1)))", 'sympy_rhs': "Mul(Symbol('pdg0004621'), Add(Mul(Symbol('pdg0004621'), sin(Symbol('pdg0001464'))), cos(Symbol('pdg0001464'))))"}, '9492920340': {'latex_lhs': 'y', 'latex_rhs': '\\cos(x)+i \\sin(x)', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001452')", 'sympy_rhs': "Add(Mul(Symbol('pdg0004621'), sin(Symbol('pdg0001464'))), cos(Symbol('pdg0001464')))"}, '9848292229': {'latex_lhs': 'dy', 'latex_rhs': 'y i dx', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0005842')", 'sympy_rhs': "Mul(Symbol('pdg0001452'), Symbol('pdg0004621'), Symbol('pdg0009199'))"}, '9848294829': {'latex_lhs': '\\\\frac{d}{dx} y', 'latex_rhs': 'y i', 'latex_relation': '=', 'sympy_lhs': "Derivative(Symbol('pdg0001452'), Tuple(Symbol('pdg0001464'), Integer(1)))", 'sympy_rhs': "Mul(Symbol('pdg0001452'), Symbol('pdg0004621'))"}}
EXPECTED_STEPS=[
 ('111649',{'9492920340','9429829482'}),('111260',{'9429829482','9482984922'}),
 ('111634',{'9492920340','9482984922','9848294829'}),('111182',{'9848294829','9848292229'}),
 ('111975',{'9848292229','9482113948'}),('111132',{'9482113948','9482943948'}),
 ('111132',{'9482943948','4928239482'}),('111268',{'4928239482','4923339482'}),
 ('111721',{'4923339482','9482923849'}),('111634',{'9482923849','9492920340','4938429483'})]
FEEDS=[["Symbol('pdg0001464')"],["Symbol('pdg0004621')"],[],["Symbol('pdg0009199')"],
       ["Symbol('pdg0001452')"],["Symbol('pdg0001452')"],["Symbol('pdg0001464')"],[],["Symbol('pdg0002718')"],[]]
MISSING={'4938429483'}



def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=10 or len(bindings)!=22 or any(b['status']!='active' for b in bindings):
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
        raise ValueError('Ten stored equations and one recovered equation required')
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
    result=dict(approved=False,source_nodes=10,source_bindings=22,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=verify_proof(build_proof())['corrections'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_euler_formula_source.py','sciona/physics_ingest/euler_formula_proof.py']})

    historical=json.loads((root/'docs/reviews/euler_formula_execution.json').read_text())['source_proof']
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
