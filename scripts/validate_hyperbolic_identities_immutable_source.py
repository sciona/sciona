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
from sciona.physics_ingest.hyperbolic_identities_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1038566242': {'sympy_lhs': "sinh(Symbol('pdg0001464'))", 'sympy_rhs': "Add(Mul(Rational(1, 2), exp(Symbol('pdg0001464'))), Mul(Integer(-1), Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464')))))", 'latex_lhs': '\\sinh x', 'latex_rhs': '\\\\frac{\\exp(x) - \\exp(-x)}{2}', 'latex_relation': '='}, '1128605625': {'sympy_lhs': "Add(Pow(tanh(Symbol('pdg0001464')), Integer(2)), Pow(sech(Symbol('pdg0001464')), Integer(2)))", 'sympy_rhs': "Add(Mul(Pow(Add(exp(Symbol('pdg0001464')), Mul(Integer(-1), exp(Mul(Integer(-1), Symbol('pdg0001464'))))), Integer(2)), Pow(Add(exp(Symbol('pdg0001464')), exp(Mul(Integer(-1), Symbol('pdg0001464')))), Integer(-2))), Mul(Integer(4), Pow(Add(exp(Symbol('pdg0001464')), exp(Mul(Integer(-1), Symbol('pdg0001464')))), Integer(-2))))", 'latex_lhs': '{\\\\rm sech}^2\\ x + \\\\tanh^2(x)', 'latex_rhs': '\\\\frac{4}{\\left(\\exp(x)+\\exp(-x)\\\\right)^2} + \\\\frac{\\left(\\exp(x)-\\exp(-x)\\\\right)^2}{\\left(\\exp(x)+\\exp(-x)\\\\right)^2}', 'latex_relation': '='}, '2103023049': {'sympy_lhs': "sin(Symbol('pdg0001464'))", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0004621'), Integer(-1)), Add(exp(Mul(Symbol('pdg0001464'), Symbol('pdg0004621'))), Mul(Integer(-1), exp(Mul(Integer(-1), Symbol('pdg0001464'), Symbol('pdg0004621'))))))", 'latex_lhs': '\\sin(x)', 'latex_rhs': '\\\\frac{1}{2i}\\left(\\exp(i x)-\\exp(-i x) \\\\right)', 'latex_relation': '='}, '2121790783': {'sympy_lhs': "Pow(tanh(Symbol('pdg0001464')), Integer(2))", 'sympy_rhs': "Mul(Pow(Add(exp(Symbol('pdg0001464')), Mul(Integer(-1), exp(Mul(Integer(-1), Symbol('pdg0001464'))))), Integer(2)), Pow(Add(exp(Symbol('pdg0001464')), exp(Mul(Integer(-1), Symbol('pdg0001464')))), Integer(-2)))", 'latex_lhs': '\\\\tanh^2(x)', 'latex_rhs': '\\\\frac{ \\left(\\exp(x)-\\exp(-x)\\\\right)^2}{\\left(\\exp(x)+\\exp(-x)\\\\right)^2}', 'latex_relation': '='}, '2762326680': {'sympy_lhs': "Add(Mul(Integer(-1), Pow(sinh(Symbol('pdg0001464')), Integer(2))), Pow(cosh(Symbol('pdg0001464')), Integer(2)))", 'sympy_rhs': 'Integer(1)', 'latex_lhs': '\\cosh^2 x - \\sinh^2 x', 'latex_rhs': '\\\\frac{1}{4}\\left( \\exp(2x)+1+1+\\exp(-2x) - \\left(\\exp(2x)-1-1+\\exp(-2x)\\\\right) \\\\right)', 'latex_relation': '='}, '2902772962': {'sympy_lhs': "tanh(Symbol('pdg0001464'))", 'sympy_rhs': "Mul(Add(Mul(Rational(1, 2), exp(Symbol('pdg0001464'))), Mul(Integer(-1), Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464'))))), Pow(cosh(Symbol('pdg0001464')), Integer(-1)))", 'latex_lhs': '\\\\tanh(x)', 'latex_rhs': '\\\\frac{\\\\frac{1}{2}\\left( \\exp(x)-\\exp(-x) \\\\right)}{\\cosh(x)}', 'latex_relation': '='}, '3868998312': {'sympy_lhs': "Pow(sech(Symbol('pdg0001464')), Integer(2))", 'sympy_rhs': "Mul(Integer(4), Pow(Add(exp(Symbol('pdg0001464')), exp(Mul(Integer(-1), Symbol('pdg0001464')))), Integer(-2)))", 'latex_lhs': '{\\\\rm sech}^2\\ x', 'latex_rhs': '\\\\frac{4}{\\left(\\exp(x)+\\exp(-x)\\\\right)^2}', 'latex_relation': '='}, '4166155526': {'sympy_lhs': "sech(Symbol('pdg0001464'))", 'sympy_rhs': "Mul(Integer(2), Pow(Add(exp(Symbol('pdg0001464')), exp(Mul(Integer(-1), Symbol('pdg0001464')))), Integer(-1)))", 'latex_lhs': '{\\\\rm sech}\\ x', 'latex_rhs': '\\\\frac{2}{\\exp(x)+\\exp(-x)}', 'latex_relation': '='}, '4585932229': {'sympy_lhs': "cos(Symbol('pdg0001464'))", 'sympy_rhs': "Add(Mul(Rational(1, 2), exp(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))), Mul(Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464'), Symbol('pdg0004621')))))", 'latex_lhs': '\\cos(x)', 'latex_rhs': '\\\\frac{1}{2}\\left(\\exp(i x)+\\exp(-i x) \\\\right)', 'latex_relation': '='}, '4830221561': {'sympy_lhs': "Add(Pow(tanh(Symbol('pdg0001464')), Integer(2)), Pow(sech(Symbol('pdg0001464')), Integer(2)))", 'sympy_rhs': "Mul(Pow(Add(exp(Symbol('pdg0001464')), exp(Mul(Integer(-1), Symbol('pdg0001464')))), Integer(-2)), Add(exp(Mul(Integer(2), Symbol('pdg0001464'))), Integer(2), exp(Mul(Integer(-1), Integer(2), Symbol('pdg0001464')))))", 'latex_lhs': '{\\\\rm sech}^2\\ x + \\\\tanh^2(x)', 'latex_rhs': '\\\\frac{4+\\left(\\exp(2x)-1-1+\\exp(-2x)\\\\right)}{\\left(\\exp(x)+\\exp(-x)\\\\right)^2}', 'latex_relation': '='}, '4872163189': {'sympy_lhs': "tanh(Symbol('pdg0001464'))", 'sympy_rhs': "Mul(sinh(Symbol('pdg0001464')), Pow(cosh(Symbol('pdg0001464')), Integer(-1)))", 'latex_lhs': '\\\\tanh(x)', 'latex_rhs': '\\\\frac{\\sinh(x)}{\\cosh(x)}', 'latex_relation': '='}, '4878728014': {'sympy_lhs': "sin(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0004621'), Integer(-1)), Add(Mul(Integer(-1), exp(Symbol('pdg0001464'))), exp(Mul(Integer(-1), Symbol('pdg0001464')))))", 'latex_lhs': '\\sin(i x)', 'latex_rhs': '\\\\frac{1}{2i}\\left(\\exp(-x) - \\exp(x) \\\\right)', 'latex_relation': '='}, '5323719091': {'sympy_lhs': "Mul(Symbol('pdg0004621'), sinh(Symbol('pdg0001464')))", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0004621'), Integer(-1)), Add(Mul(Integer(-1), exp(Symbol('pdg0001464'))), exp(Mul(Integer(-1), Symbol('pdg0001464')))))", 'latex_lhs': 'i \\sinh x', 'latex_rhs': '\\\\frac{1}{2i} \\left( \\exp(-x) - \\exp(x) \\\\right)', 'latex_relation': '='}, '5349669879': {'sympy_lhs': "tanh(Symbol('pdg0001464'))", 'sympy_rhs': "Mul(Add(exp(Symbol('pdg0001464')), Mul(Integer(-1), exp(Mul(Integer(-1), Symbol('pdg0001464'))))), Pow(Add(exp(Symbol('pdg0001464')), exp(Mul(Integer(-1), Symbol('pdg0001464')))), Integer(-1)))", 'latex_lhs': '\\\\tanh(x)', 'latex_rhs': '\\\\frac{ \\exp(x)-\\exp(-x)}{\\exp(x)+\\exp(-x)}', 'latex_relation': '='}, '5866629429': {'sympy_lhs': "Add(Pow(tanh(Symbol('pdg0001464')), Integer(2)), Pow(sech(Symbol('pdg0001464')), Integer(2)))", 'sympy_rhs': 'Integer(1)', 'latex_lhs': '{\\\\rm sech}^2\\ x + \\\\tanh^2(x)', 'latex_rhs': '1', 'latex_relation': '='}, '6031385191': {'sympy_lhs': "Pow(sinh(Symbol('pdg0001464')), Integer(2))", 'sympy_rhs': "Pow(Add(Mul(Rational(1, 2), exp(Symbol('pdg0001464'))), Mul(Integer(-1), Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464'))))), Integer(2))", 'latex_lhs': '\\sinh^2 x', 'latex_rhs': '\\left(\\\\frac{\\exp(x) - \\exp(-x)}{2}\\\\right)\\left(\\\\frac{\\exp(x) - \\exp(-x)}{2}\\\\right)', 'latex_relation': '='}, '6404535647': {'sympy_lhs': "cosh(Symbol('pdg0001464'))", 'sympy_rhs': "Add(Mul(Rational(1, 2), exp(Symbol('pdg0001464'))), Mul(Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464')))))", 'latex_lhs': '\\cosh x', 'latex_rhs': '\\\\frac{\\exp(x) + \\exp(-x)}{2}', 'latex_relation': '='}, '7731226616': {'sympy_lhs': "sech(Symbol('pdg0001464'))", 'sympy_rhs': "Pow(cosh(Symbol('pdg0001464')), Integer(-1))", 'latex_lhs': '{\\\\rm sech}\\ x', 'latex_rhs': '\\\\frac{1}{\\cosh x}', 'latex_relation': '='}, '8418527415': {'sympy_lhs': "sin(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Mul(Symbol('pdg0004621'), sinh(Symbol('pdg0001464')))", 'latex_lhs': '\\sin(i x)', 'latex_rhs': 'i \\sinh(x)', 'latex_relation': '='}, '8532702080': {'sympy_lhs': "Pow(cosh(Symbol('pdg0001464')), Integer(2))", 'sympy_rhs': "Pow(Add(Mul(Rational(1, 2), exp(Symbol('pdg0001464'))), Mul(Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464'))))), Integer(2))", 'latex_lhs': '\\cosh^2 x', 'latex_rhs': '\\left(\\\\frac{\\exp(x) + \\exp(-x)}{2}\\\\right)\\left(\\\\frac{\\exp(x) + \\exp(-x)}{2}\\\\right)', 'latex_relation': '='}, '8563535636': {'sympy_lhs': "Add(Mul(Integer(-1), Pow(sinh(Symbol('pdg0001464')), Integer(2))), Pow(cosh(Symbol('pdg0001464')), Integer(2)))", 'sympy_rhs': "Add(Pow(Add(Mul(Rational(1, 2), exp(Symbol('pdg0001464'))), Mul(Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464'))))), Integer(2)), Mul(Integer(-1), Rational(1, 4), Pow(Add(exp(Symbol('pdg0001464')), Mul(Integer(-1), exp(Mul(Integer(-1), Symbol('pdg0001464'))))), Integer(2))))", 'latex_lhs': '\\cosh^2 x - \\sinh^2 x', 'latex_rhs': '\\left(\\\\frac{\\exp(x) + \\exp(-x)}{2}\\\\right)\\left(\\\\frac{\\exp(x) + \\exp(-x)}{2}\\\\right) - \\left(\\\\frac{\\exp(x) - \\exp(-x)}{2}\\\\right)\\left(\\\\frac{\\exp(x) - \\exp(-x)}{2}\\\\right)', 'latex_relation': '='}, '8651044341': {'sympy_lhs': "cos(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "Add(Mul(Rational(1, 2), exp(Symbol('pdg0001464'))), Mul(Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464')))))", 'latex_lhs': '\\cos(i x)', 'latex_rhs': '\\\\frac{1}{2} \\left( \\exp(-x) + \\exp(x) \\\\right)', 'latex_relation': '='}, '8747785338': {'sympy_lhs': "cos(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))", 'sympy_rhs': "cosh(Symbol('pdg0001464'))", 'latex_lhs': '\\cos(i x)', 'latex_rhs': '\\cosh(x)', 'latex_relation': '='}, '9413609246': {'sympy_lhs': "Add(Mul(Integer(-1), Pow(sinh(Symbol('pdg0001464')), Integer(2))), Pow(cosh(Symbol('pdg0001464')), Integer(2)))", 'sympy_rhs': 'Integer(1)', 'latex_lhs': '\\cosh^2 x - \\sinh^2 x', 'latex_rhs': '1', 'latex_relation': '='}}
EXPECTED_STEPS = [('111253', {'1038566242', '6031385191'}), ('111253', {'6404535647', '8532702080'}), ('111222', {'8563535636', '6031385191', '8532702080'}), ('111457', {'2762326680', '8563535636'}), ('111457', {'2762326680', '9413609246'}), ('111886', {'4878728014', '2103023049'}), ('111355', {'4878728014', '8418527415', '5323719091'}), ('111182', {'1038566242', '5323719091'}), ('111886', {'4585932229', '8651044341'}), ('111355', {'6404535647', '8747785338', '8651044341'}), ('111556', {'6404535647', '7731226616', '4166155526'}), ('111556', {'1038566242', '4872163189', '2902772962'}), ('111556', {'6404535647', '2902772962', '5349669879'}), ('111253', {'2121790783', '5349669879'}), ('111253', {'3868998312', '4166155526'}), ('111980', {'1128605625', '3868998312', '2121790783'}), ('111457', {'1128605625', '4830221561'}), ('111457', {'4830221561', '5866629429'})]
FEEDS = [[], [], [], [], [], ["Symbol('pdg0001464')", "Mul(Symbol('pdg0004621'), Symbol('pdg0001464'))"], [], ["Symbol('pdg0004621')"], ["Symbol('pdg0001464')", "Mul(Symbol('pdg0004621'), Symbol('pdg0001464'))"], [], [], [], [], [], [], [], [], []]
MISSING = {'4585932229'}


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=18 or len(bindings)!=43 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=18,source_bindings=43,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Use mathematical imaginary unit I, not an unconstrained named source symbol.', 'Retain explicit LaTex algebraic expansion where the stored step4 AST is already simplified to one.', 'Self-products reuse the same equation twice although projected binding inventory contains it once.', 'Replay step8 before step7 to satisfy the imaginary-sine comparison dependency.', 'Use entire complex sine/cosine exponential definitions for arguments I*x; a real-only Euler certificate cannot simply be substituted at complex input.', 'Restrict x to finite real dimensionless values so cosh(x)>0 and all quotient identities are defined.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_hyperbolic_identities_source.py','sciona/physics_ingest/hyperbolic_identities_proof.py']})


    historical=json.loads((root/'docs/reviews/hyperbolic_identities_execution.json').read_text())['source_proof']
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
