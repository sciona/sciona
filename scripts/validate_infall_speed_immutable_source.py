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
from sciona.physics_ingest.infall_speed_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1114820451': {'sympy_lhs': "Symbol('pdg0006191')", 'sympy_rhs': "Symbol('pdg0005734')", 'latex_lhs': 'W_{\\\\rm by\\ system}', 'latex_rhs': '\\Delta KE', 'latex_relation': '='}, '2005061870': {'sympy_lhs': "Function('pdg0001357')(Symbol('pdg0002530'))", 'sympy_rhs': "Mul(Pow(Integer(2), Rational(1, 2)), Pow(Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0004851'), Symbol('pdg0006277')), Rational(1, 2)))", 'latex_lhs': 'v(r)', 'latex_rhs': '\\sqrt{\\\\frac{2 G m_2}{r}}', 'latex_relation': '='}, '2061086175': {'sympy_lhs': "Symbol('pdg0009372')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0005022'), Symbol('pdg0006277'), Function('pdg0004851')(Mul(Integer(-1), Pow(Symbol('pdg0002530'), Integer(-1)))))", 'latex_lhs': 'W_{\\\\rm to\\ system}', 'latex_rhs': '-G m_1 m_2 \\left(\\\\frac{-1}{r} - \\\\frac{-1}{\\infty}\\\\right)', 'latex_relation': '='}, '2907404069': {'sympy_lhs': "Symbol('pdg0006191')", 'sympy_rhs': "Symbol('pdg0009372')", 'latex_lhs': 'W_{\\\\rm by\\ system}', 'latex_rhs': 'W_{\\\\rm to\\ system}', 'latex_relation': '='}, '2924222857': {'sympy_lhs': "Symbol('pdg0001934')", 'sympy_rhs': "Symbol('pdg0001357')", 'latex_lhs': 'v_{\\\\rm initial}', 'latex_rhs': 'v(r=\\infty)', 'latex_relation': '='}, '2998709778': {'sympy_lhs': "Symbol('pdg0001934')", 'sympy_rhs': 'Integer(0)', 'latex_lhs': 'v_{\\\\rm initial}', 'latex_rhs': '0', 'latex_relation': '='}, '3214170322': {'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': '', 'latex_lhs': 'v(r=\\infty)', 'latex_rhs': '0', 'latex_relation': '='}, '3566149658': {'sympy_lhs': "Symbol('pdg0009372')", 'sympy_rhs': "Integral(Mul(Integer(-1), Pow(Symbol('pdg0004037'), Integer(-2)), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277')), Tuple(Symbol('pdg0004037'), oo, Symbol('pdg0002530')))", 'latex_lhs': 'W_{\\\\rm to\\ system}', 'latex_rhs': '\\int_{\\infty}^r \\\\frac{-G m_1 m_2}{x^2} dx', 'latex_relation': '='}, '4393670960': {'sympy_lhs': "Symbol('pdg0009372')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'))", 'latex_lhs': 'W_{\\\\rm to\\ system}', 'latex_rhs': '\\\\frac{G m_1 m_2}{r}', 'latex_relation': '='}, '4947831649': {'sympy_lhs': "Mul(Rational(1, 2), Symbol('pdg0005022'), Pow(Symbol('pdg0008909'), Integer(2)))", 'sympy_rhs': "Symbol('pdg0009372')", 'latex_lhs': '\\\\frac{1}{2} m_1 v_{\\\\rm final}^2', 'latex_rhs': 'W_{\\\\rm to\\ system}', 'latex_relation': '='}, '5596822289': {'sympy_lhs': "Symbol('pdg0009372')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'))", 'latex_lhs': 'W_{\\\\rm to\\ system}', 'latex_rhs': '-G m_1 m_2 \\left(\\left.\\\\frac{-1}{x}\\\\right|^r_{\\infty}\\\\right)', 'latex_relation': '='}, '5693047217': {'sympy_lhs': "Symbol('pdg0008909')", 'sympy_rhs': "Mul(Integer(-1), Pow(Integer(2), Rational(1, 2)), Pow(Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0004851'), Symbol('pdg0006277')), Rational(1, 2)))", 'latex_lhs': 'v_{\\\\rm final}', 'latex_rhs': '-\\sqrt{\\\\frac{2 G m_2}{r}}', 'latex_relation': '='}, '5779256336': {'sympy_lhs': "Symbol('pdg0006191')", 'sympy_rhs': "Add(Mul(Integer(-1), Symbol('pdg0004121')), Symbol('pdg0005340'))", 'latex_lhs': 'W_{\\\\rm by\\ system}', 'latex_rhs': 'KE_{\\\\rm final} - KE_{\\\\rm initial}', 'latex_relation': '='}, '5846639423': {'sympy_lhs': "Symbol('pdg0008909')", 'sympy_rhs': "Mul(Pow(Integer(2), Rational(1, 2)), Pow(Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0004851'), Symbol('pdg0006277')), Rational(1, 2)))", 'latex_lhs': 'v_{\\\\rm final}', 'latex_rhs': '\\sqrt{\\\\frac{2 G m_2}{r}}', 'latex_relation': '='}, '5850144586': {'sympy_lhs': "Symbol('pdg0006191')", 'sympy_rhs': "Symbol('pdg0005340')", 'latex_lhs': 'W_{\\\\rm by\\ system}', 'latex_rhs': 'KE_{\\\\rm final}', 'latex_relation': '='}, '5902985919': {'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0004037'), Integer(-1)), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'))", 'latex_lhs': '\\vec{F}', 'latex_rhs': 'G \\\\frac{m_1 m_2}{x^2} \\hat{x}', 'latex_relation': '='}, '6091977310': {'sympy_lhs': "Symbol('pdg0004121')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0001934'), Integer(2)), Symbol('pdg0005022'))", 'latex_lhs': 'KE_{\\\\rm initial}', 'latex_rhs': '\\\\frac{1}{2} m_1 v_{\\\\rm initial}^2', 'latex_relation': '='}, '6892595652': {'sympy_lhs': "Mul(Rational(1, 2), Symbol('pdg0005022'), Pow(Symbol('pdg0008909'), Integer(2)))", 'sympy_rhs': "Mul(Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'))", 'latex_lhs': '\\\\frac{1}{2} m_1 v_{\\\\rm final}^2', 'latex_rhs': '\\\\frac{G m_1 m_2}{r}', 'latex_relation': '='}, '7112646057': {'sympy_lhs': "Pow(Symbol('pdg0008909'), Integer(2))", 'sympy_rhs': "Mul(Integer(2), Pow(Symbol('pdg0002530'), Integer(-1)), Symbol('pdg0004851'), Symbol('pdg0006277'))", 'latex_lhs': 'v_{\\\\rm final}^2', 'latex_rhs': '\\\\frac{2 G m_2}{r}', 'latex_relation': '='}, '7882872592': {'sympy_lhs': "Symbol('pdg0009372')", 'sympy_rhs': "Integral(Function('Dot')(Symbol('pdg0006777'), Symbol('pdg0002530')), Tuple(Symbol('pdg0002530'), oo, Symbol('pdg0002530')))", 'latex_lhs': 'W_{\\\\rm to\\ system}', 'latex_rhs': '\\int_{\\infty}^r \\vec{F}\\cdot d\\vec{r}', 'latex_relation': '='}, '8049905441': {'sympy_lhs': "Symbol('pdg0005734')", 'sympy_rhs': "Add(Mul(Integer(-1), Symbol('pdg0004121')), Symbol('pdg0005340'))", 'latex_lhs': '\\Delta KE', 'latex_rhs': 'KE_{\\\\rm final} - KE_{\\\\rm initial}', 'latex_relation': '='}, '8357234146': {'sympy_lhs': "Symbol('pdg0004929')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0001357'), Integer(2)), Symbol('pdg0005156'))", 'latex_lhs': 'KE', 'latex_rhs': '\\\\frac{1}{2} m v^2', 'latex_relation': '='}, '8405272745': {'sympy_lhs': "Symbol('pdg0009372')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'), Integral(Pow(Symbol('pdg0004037'), Integer(-2)), Tuple(Symbol('pdg0004037'), oo, Symbol('pdg0002530'))))", 'latex_lhs': 'W_{\\\\rm to\\ system}', 'latex_rhs': '-G m_1 m_2\\int_{\\infty}^r \\\\frac{1}{x^2} dx', 'latex_relation': '='}, '8552710882': {'sympy_lhs': "Symbol('pdg0005340')", 'sympy_rhs': "Mul(Rational(1, 2), Symbol('pdg0005022'), Pow(Symbol('pdg0008909'), Integer(2)))", 'latex_lhs': 'KE_{\\\\rm final}', 'latex_rhs': '\\\\frac{1}{2} m_1 v_{\\\\rm final}^2', 'latex_relation': '='}, '9081138616': {'sympy_lhs': "Symbol('pdg0006191')", 'sympy_rhs': "Mul(Rational(1, 2), Symbol('pdg0005022'), Pow(Symbol('pdg0008909'), Integer(2)))", 'latex_lhs': 'W_{\\\\rm by\\ system}', 'latex_rhs': '\\\\frac{1}{2} m_1 v_{\\\\rm final}^2', 'latex_relation': '='}, '9510328252': {'sympy_lhs': "Symbol('pdg0004121')", 'sympy_rhs': 'Integer(0)', 'latex_lhs': 'KE_{\\\\rm initial}', 'latex_rhs': '0', 'latex_relation': '='}}
EXPECTED_STEPS = [('111556', {'3566149658', '5902985919', '7882872592'}), ('111457', {'3566149658', '8405272745'}), ('111662', {'5596822289', '8405272745'}), ('111457', {'2061086175', '5596822289'}), ('111457', {'2061086175', '4393670960'}), ('111556', {'8049905441', '1114820451', '5779256336'}), ('111236', {'8357234146', '6091977310'}), ('111236', {'8357234146', '8552710882'}), ('111556', {'2924222857', '3214170322', '2998709778'}), ('111556', {'6091977310', '9510328252', '2998709778'}), ('111556', {'5850144586', '9510328252', '5779256336'}), ('111556', {'9081138616', '5850144586', '8552710882'}), ('111556', {'9081138616', '4947831649', '2907404069'}), ('111556', {'4393670960', '6892595652', '4947831649'}), ('111182', {'7112646057', '6892595652'}), ('111524', {'5693047217', '7112646057', '5846639423'}), ('111886', {'2005061870', '5846639423'})]
FEEDS = [[], [], [], [], [], [], ["Symbol('pdg0004929')", "Symbol('pdg0004121')", "Symbol('pdg0005156')", "Symbol('pdg0005022')", "Symbol('pdg0001357')", "Symbol('pdg0001934')"], ["Symbol('pdg0004929')", "Symbol('pdg0005340')", "Symbol('pdg0005156')", "Symbol('pdg0005022')", "Symbol('pdg0001357')", "Symbol('pdg0008909')"], [], [], [], [], [], [], ["Mul(Integer(2), Pow(Symbol('pdg0005022'), Integer(-1)))"], [], ["Symbol('pdg0008909')", "Function('pdg0001357')(Symbol('pdg0002530'))"]]
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
        if len(nodes)!=17 or len(bindings)!=43 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=17,source_bindings=43,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Force is inward: radial component -G*m*M/x^2; source LaTex sign and AST power/vector information disagree.', 'Reconstruct vector work as scalar radial integral with a distinct dummy coordinate.', 'Restore missing antiderivative/boundary terms in step3 and scalar mass multiplication in step4.', 'Work-by and work-to aliases both mean gravitational work on the falling test particle, not opposite thermodynamic work conventions.', 'Restore missing zero RHS and lost infinity evaluation in boundary ASTs2924222857and3214170322.', 'Use fixed central mass and negligible test mass; rest at infinity is an asymptotic zero-energy boundary condition.', 'Step16 roots are alternative signed radial velocities; step17 selects positive speed magnitude, while inward radial velocity is negative.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_infall_speed_source.py','sciona/physics_ingest/infall_speed_proof.py']})

    historical=json.loads((root/'docs/reviews/infall_speed_execution.json').read_text())['source_proof']
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
