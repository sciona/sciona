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
from sciona.physics_ingest.langmuir_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'2114909846': {'latex_lhs': '\\\\theta_A', 'latex_rhs': '\\\\frac{[A_{\\\\rm adsorption}]}{[S_0]}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001791')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0003037'), Integer(-1)), Symbol('pdg0004940'))"}, '2168306601': {'latex_lhs': '[S_0]', 'latex_rhs': '\\left(\\\\frac{k_{\\\\rm desorption}}{k_{\\\\rm adsorption}} \\\\frac{1}{p_A} + 1\\\\right)[A_{\\\\rm adsorption}]', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0003037')", 'sympy_rhs': "Mul(Symbol('pdg0004940'), Add(Integer(1), Mul(Pow(Symbol('pdg0006850'), Integer(-1)), Symbol('pdg0008379'), Pow(Symbol('pdg0009046'), Integer(-1)))))"}, '3488423948': {'latex_lhs': 'k_{\\\\rm adsorption} p_A [S]', 'latex_rhs': 'k_{\\\\rm desorption} [A_{\\\\rm adsorption}]', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0006850'), Symbol('pdg0009046'), Symbol('pdg0009067'))", 'sympy_rhs': "Mul(Symbol('pdg0004940'), Symbol('pdg0008379'))"}, '3507029294': {'latex_lhs': 'k_{\\\\rm adsorption} p_A [S]', 'latex_rhs': 'r_{\\\\rm desorption}', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0006850'), Symbol('pdg0009046'), Symbol('pdg0009067'))", 'sympy_rhs': "Symbol('pdg0001966')"}, '3599953931': {'latex_lhs': '[S_0]', 'latex_rhs': '[S] + [A_{\\\\rm adsorption}]', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0003037')", 'sympy_rhs': "Add(Symbol('pdg0004940'), Symbol('pdg0009067'))"}, '3736177473': {'latex_lhs': 'r_{\\\\rm adsorption}', 'latex_rhs': 'k_{\\\\rm adsorption} p_A [S]', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0006687')", 'sympy_rhs': "Mul(Symbol('pdg0006850'), Symbol('pdg0009046'), Symbol('pdg0009067'))"}, '4301729661': {'latex_lhs': '[S_0]', 'latex_rhs': '\\\\frac{[A_{\\\\rm adsorption}]}{\\left( \\\\frac{k_{\\\\rm adsorption}}{k_{\\\\rm desorption}} \\\\right) p_A} + [A_{\\\\rm adsorption}]', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0003037')", 'sympy_rhs': "Add(Symbol('pdg0004940'), Mul(Symbol('pdg0004940'), Pow(Symbol('pdg0006850'), Integer(-1)), Symbol('pdg0008379'), Pow(Symbol('pdg0009046'), Integer(-1))))"}, '4689334676': {'latex_lhs': '\\\\theta_A', 'latex_rhs': '\\\\frac{K_{\\\\rm equilibrium}\\ p_A}{1+K_{\\\\rm equilibrium}\\ p_A}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001791')", 'sympy_rhs': "Mul(Symbol('pdg0004933'), Symbol('pdg0009046'), Pow(Add(Mul(Symbol('pdg0004933'), Symbol('pdg0009046')), Integer(1)), Integer(-1)))"}, '5085809757': {'latex_lhs': '\\\\frac{k_{\\\\rm adsorption}}{k_{\\\\rm desorption}}', 'latex_rhs': '\\\\frac{[A_{\\\\rm adsorption}]}{p_A [S]}', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0006850'), Pow(Symbol('pdg0008379'), Integer(-1)))", 'sympy_rhs': "Mul(Symbol('pdg0004940'), Pow(Symbol('pdg0009046'), Integer(-1)), Pow(Symbol('pdg0009067'), Integer(-1)))"}, '6240546932': {'latex_lhs': '\\\\frac{1}{K_{equilibrium}}', 'latex_rhs': '\\\\frac{k_{\\\\rm desorption}}{k_{\\\\rm adsorption}}', 'latex_relation': '=', 'sympy_lhs': "Pow(Symbol('pdg0004933'), Integer(-1))", 'sympy_rhs': "Mul(Pow(Symbol('pdg0006850'), Integer(-1)), Symbol('pdg0008379'))"}, '6457999644': {'latex_lhs': '\\\\frac{[S_0]}{[A_{\\\\rm adsorption}]}', 'latex_rhs': '\\\\frac{1}{K_{\\\\rm equilibrium}} \\\\frac{1}{p_A} + 1', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0003037'), Pow(Symbol('pdg0004940'), Integer(-1)))", 'sympy_rhs': "Add(Integer(1), Mul(Pow(Symbol('pdg0004933'), Integer(-1)), Pow(Symbol('pdg0009046'), Integer(-1))))"}, '6783009163': {'latex_lhs': 'r_{\\\\rm adsorption}', 'latex_rhs': 'r_{\\\\rm desorption}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0006687')", 'sympy_rhs': "Symbol('pdg0001966')"}, '6955192897': {'latex_lhs': 'r_{\\\\rm desorption}', 'latex_rhs': 'k_{\\\\rm desorption} [A_{\\\\rm adsorption}]', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001966')", 'sympy_rhs': "Mul(Symbol('pdg0004940'), Symbol('pdg0008379'))"}, '7267424860': {'latex_lhs': '\\\\frac{1}{\\\\theta_A}', 'latex_rhs': '\\\\frac{1+(K_{\\\\rm equilibrium}\\ p_A)}{K_{\\\\rm equilibrium}\\ p_A}', 'latex_relation': '=', 'sympy_lhs': "Pow(Symbol('pdg0001791'), Integer(-1))", 'sympy_rhs': "Mul(Pow(Symbol('pdg0004933'), Integer(-1)), Pow(Symbol('pdg0009046'), Integer(-1)), Add(Mul(Symbol('pdg0004933'), Symbol('pdg0009046')), Integer(1)))"}, '7517073655': {'latex_lhs': '[S_0]', 'latex_rhs': '\\left(\\\\frac{1}{K_{\\\\rm equilibrium}} \\\\frac{1}{p_A} + 1\\\\right)[A_{\\\\rm adsorption}]', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0003037')", 'sympy_rhs': "Mul(Symbol('pdg0004940'), Add(Integer(1), Mul(Pow(Symbol('pdg0004933'), Integer(-1)), Pow(Symbol('pdg0009046'), Integer(-1)))))"}, '7924063906': {'latex_lhs': 'K_{equilibrium}', 'latex_rhs': '\\\\frac{k_{\\\\rm adsorption}}{k_{\\\\rm desorption}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004933')", 'sympy_rhs': "Mul(Symbol('pdg0006850'), Pow(Symbol('pdg0008379'), Integer(-1)))"}, '7928111771': {'latex_lhs': '\\\\frac{1}{\\\\theta_A}', 'latex_rhs': '\\\\frac{1}{K_{\\\\rm equilibrium} p_A} + 1', 'latex_relation': '=', 'sympy_lhs': "Pow(Symbol('pdg0001791'), Integer(-1))", 'sympy_rhs': "Add(Integer(1), Mul(Pow(Symbol('pdg0004933'), Integer(-1)), Pow(Symbol('pdg0009046'), Integer(-1))))"}, '8131665171': {'latex_lhs': '\\\\frac{1}{\\\\theta_A}', 'latex_rhs': '\\\\frac{[S_0]}{[A_{\\\\rm adsorption}]}', 'latex_relation': '=', 'sympy_lhs': "Pow(Symbol('pdg0001791'), Integer(-1))", 'sympy_rhs': "Mul(Symbol('pdg0003037'), Pow(Symbol('pdg0004940'), Integer(-1)))"}, '9562264720': {'latex_lhs': '[S]', 'latex_rhs': '\\\\frac{k_{\\\\rm desorption} [A_{\\\\rm adsorption}]}{k_{\\\\rm adsorption} p_A}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0009067')", 'sympy_rhs': "Mul(Symbol('pdg0004940'), Pow(Symbol('pdg0006850'), Integer(-1)), Symbol('pdg0008379'), Pow(Symbol('pdg0009046'), Integer(-1)))"}}
EXPECTED_STEPS = [('111556', {'6783009163', '3736177473', '3507029294'}), ('111556', {'6955192897', '3488423948', '3507029294'}), ('111975', {'3488423948', '9562264720'}), ('111556', {'3599953931', '9562264720', '4301729661'}), ('111432', {'2168306601', '4301729661'}), ('111975', {'3488423948', '5085809757'}), ('111483', {'6240546932', '7924063906'}), ('111634', {'6240546932', '7517073655', '2168306601'}), ('111975', {'6457999644', '7517073655'}), ('111483', {'8131665171', '2114909846'}), ('111634', {'6457999644', '7928111771', '8131665171'}), ('111886', {'7267424860', '7928111771'}), ('111483', {'7267424860', '4689334676'})]
FEEDS = [[], [], ["Mul(Symbol('pdg0006850'), Symbol('pdg0009046'))"], [], ["Symbol('pdg0004940')"], ["Mul(Symbol('pdg0009067'), Symbol('pdg0009046'))"], ['Integer(-1)'], [], ["Symbol('pdg0004940')"], ['Integer(-1)'], [], ['Integer(1)', "Mul(Pow(Mul(Symbol('pdg0004933'), Symbol('pdg0009046')), Integer(-1)), Mul(Symbol('pdg0004933'), Symbol('pdg0009046')))"], ['Integer(-1)']]
MISSING = set()


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash']!=SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=13 or len(bindings)!=31 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=13,source_bindings=31,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Step6 divides by kd*p*S, not only the displayed p*S feed.', 'Reciprocal source steps require positive pressure and positive occupied/vacant densities.', 'Verify zero-pressure endpoint independently using original equilibrium linear system.', 'K=ka/kd is inverse pressure, not dimensionless, under pressure-based kinetics.', 'Fixed-temperature single-species independent monolayer sites; no transient or competitive model.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_langmuir_source.py','sciona/physics_ingest/langmuir_proof.py']})

    historical=json.loads((root/'docs/reviews/langmuir_execution.json').read_text())['source_proof']
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
