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
from sciona.physics_ingest.wave_interference_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1020854560': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Mul(Add(Symbol('pdg0004453'), Symbol('pdg0004698')), conjugate(Add(Symbol('pdg0004453'), Symbol('pdg0004698'))))", 'latex_lhs': 'I', 'latex_rhs': '(A + B)(A + B)^*', 'latex_relation': '='}, '1172039918': {'sympy_lhs': "Symbol('pdg0008251')", 'sympy_rhs': "Mul(Integer(4), Pow(Abs(Symbol('pdg0004453')), Integer(2)))", 'latex_lhs': 'I_{\\\\rm coherent}', 'latex_rhs': '4 |A|^2', 'latex_relation': '='}, '1357848476': {'sympy_lhs': "Symbol('pdg0004453')", 'sympy_rhs': "Mul(exp(Mul(Symbol('pdg0001575'), Symbol('pdg0004621'))), Abs(Symbol('pdg0004453')))", 'latex_lhs': 'A', 'latex_rhs': '|A| \\exp(i \\\\theta)', 'latex_relation': '='}, '1525861537': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Add(Mul(Symbol('pdg0004453'), conjugate(Symbol('pdg0004698'))), Mul(Symbol('pdg0004698'), conjugate(Symbol('pdg0004453'))), Pow(Abs(Symbol('pdg0004453')), Integer(2)), Pow(Abs(Symbol('pdg0004698')), Integer(2)))", 'latex_lhs': 'I', 'latex_rhs': '|A|^2 + |B|^2 + A B^* + B A^*', 'latex_relation': '='}, '1928085940': {'sympy_lhs': "Symbol('pdg0003192')", 'sympy_rhs': '', 'latex_lhs': 'Z^*', 'latex_rhs': '|Z| \\exp( -i \\\\theta )', 'latex_relation': '='}, '2236639474': {'sympy_lhs': "Pow(Add(Symbol('pdg0004453'), Symbol('pdg0004698')), Integer(2))", 'sympy_rhs': "Pow(Abs(Add(Symbol('pdg0004453'), Symbol('pdg0004698'))), Integer(2))", 'latex_lhs': '(A + B)(A + B)^*', 'latex_rhs': '|A + B|^2', 'latex_relation': '='}, '2700934933': {'sympy_lhs': "Mul(Integer(2), cos(Symbol('pdg0001464')))", 'sympy_rhs': "Add(exp(Mul(Symbol('pdg0004621'), Add(Symbol('pdg0001575'), Mul(Integer(-1), Symbol('pdg0008586'))))), exp(Mul(Integer(-1), Symbol('pdg0004621'), Add(Symbol('pdg0001575'), Mul(Integer(-1), Symbol('pdg0008586'))))))", 'latex_lhs': '2 \\cos(x)', 'latex_rhs': '\\left( \\exp(i (\\\\theta - \\phi)) + \\exp(-i (\\\\theta - \\phi)) \\\\right)', 'latex_relation': '='}, '2719691582': {'sympy_lhs': "Abs(Symbol('pdg0004453'))", 'sympy_rhs': "Abs(Symbol('pdg0004698'))", 'latex_lhs': '|A|', 'latex_rhs': '|B|', 'latex_relation': '='}, '3060393541': {'sympy_lhs': "Symbol('pdg0002435')", 'sympy_rhs': "Mul(Integer(2), Pow(Abs(Symbol('pdg0004453')), Integer(2)))", 'latex_lhs': 'I_{\\\\rm incoherent}', 'latex_rhs': '2|A|^2', 'latex_relation': '='}, '3085575328': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Add(Pow(Abs(Symbol('pdg0004453')), Integer(2)), Pow(Abs(Symbol('pdg0004698')), Integer(2)), Mul(exp(Mul(Integer(-1), Symbol('pdg0004621'), Add(Symbol('pdg0001575'), Mul(Integer(-1), Symbol('pdg0008586'))))), Abs(Mul(Symbol('pdg0004453'), Symbol('pdg0004698'), Abs(Add(Mul(exp(Mul(Symbol('pdg0004621'), Add(Symbol('pdg0001575'), Mul(Integer(-1), Symbol('pdg0008586'))))), Abs(Symbol('pdg0004698'))), Abs(Symbol('pdg0004453'))))))))", 'latex_lhs': 'I', 'latex_rhs': '|A|^2 + |B|^2 + |A| |B| \\exp(i (\\\\theta - \\phi)) + |A| |B| \\exp(-i (\\\\theta - \\phi))', 'latex_relation': '='}, '3350830826': {'sympy_lhs': "Symbol('pdg0003192')", 'sympy_rhs': '', 'latex_lhs': 'Z Z^*', 'latex_rhs': '|Z|^2', 'latex_relation': '='}, '3660957533': {'sympy_lhs': "cos(Symbol('pdg0001464'))", 'sympy_rhs': "Add(Mul(Rational(1, 2), exp(Mul(Symbol('pdg0004621'), Add(Symbol('pdg0001575'), Mul(Integer(-1), Symbol('pdg0008586')))))), Mul(Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0004621'), Add(Symbol('pdg0001575'), Mul(Integer(-1), Symbol('pdg0008586')))))))", 'latex_lhs': '\\cos(x)', 'latex_rhs': '\\\\frac{1}{2} \\left( \\exp(i (\\\\theta - \\phi)) + \\exp(-i (\\\\theta - \\phi)) \\\\right)', 'latex_relation': '='}, '4075539836': {'sympy_lhs': "Mul(Symbol('pdg0004453'), conjugate(Symbol('pdg0004453')))", 'sympy_rhs': "Pow(Abs(Symbol('pdg0004453')), Integer(2))", 'latex_lhs': 'A A^*', 'latex_rhs': '|A|^2', 'latex_relation': '='}, '4182362050': {'sympy_lhs': "Symbol('pdg0003192')", 'sympy_rhs': "Mul(exp(Mul(Symbol('pdg0001575'), Symbol('pdg0004621'))), Abs(Symbol('pdg0003192')))", 'latex_lhs': 'Z', 'latex_rhs': '|Z| \\exp( i \\\\theta )', 'latex_relation': '='}, '4192519596': {'sympy_lhs': "Symbol('pdg0004698')", 'sympy_rhs': "Mul(exp(Mul(Symbol('pdg0004621'), Symbol('pdg0008586'))), Abs(Symbol('pdg0004698')))", 'latex_lhs': 'B', 'latex_rhs': '|B| \\exp(i \\phi)', 'latex_relation': '='}, '4504256452': {'sympy_lhs': "conjugate(Symbol('pdg0004698'))", 'sympy_rhs': "Mul(exp(Mul(Integer(-1), Symbol('pdg0004621'), Symbol('pdg0008586'))), Abs(Symbol('pdg0004698')))", 'latex_lhs': 'B^*', 'latex_rhs': '|B| \\exp(-i \\phi)', 'latex_relation': '='}, '4585932229': {'sympy_lhs': "cos(Symbol('pdg0001464'))", 'sympy_rhs': "Add(Mul(Rational(1, 2), exp(Mul(Symbol('pdg0001464'), Symbol('pdg0004621')))), Mul(Rational(1, 2), exp(Mul(Integer(-1), Symbol('pdg0001464'), Symbol('pdg0004621')))))", 'latex_lhs': '\\cos(x)', 'latex_rhs': '\\\\frac{1}{2}\\left(\\exp(i x)+\\exp(-i x) \\\\right)', 'latex_relation': '='}, '5125940051': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Add(Mul(Symbol('pdg0004453'), conjugate(Symbol('pdg0004698'))), Mul(Symbol('pdg0004698'), conjugate(Symbol('pdg0004453'))), Mul(Symbol('pdg0004698'), conjugate(Symbol('pdg0004698'))), Pow(Abs(Symbol('pdg0004453')), Integer(2)))", 'latex_lhs': 'I', 'latex_rhs': '|A|^2 + B B^* + A B^* + B A^*', 'latex_relation': '='}, '6240206408': {'sympy_lhs': "Symbol('pdg0002435')", 'sympy_rhs': "Add(Pow(Abs(Symbol('pdg0004453')), Integer(2)), Pow(Abs(Symbol('pdg0004698')), Integer(2)))", 'latex_lhs': 'I_{\\\\rm incoherent}', 'latex_rhs': '|A|^2 + |B|^2', 'latex_relation': '='}, '6306552185': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Mul(Add(Symbol('pdg0004453'), Symbol('pdg0004698')), Add(conjugate(Symbol('pdg0004453')), conjugate(Symbol('pdg0004698'))))", 'latex_lhs': 'I', 'latex_rhs': '(A + B)(A^* + B^*)', 'latex_relation': '='}, '6529793063': {'sympy_lhs': "Symbol('pdg0002435')", 'sympy_rhs': "Mul(Integer(2), Pow(Abs(Symbol('pdg0004453')), Integer(2)))", 'latex_lhs': 'I_{\\\\rm incoherent}', 'latex_rhs': '|A|^2 + |A|^2', 'latex_relation': '='}, '6555185548': {'sympy_lhs': "conjugate(Symbol('pdg0004453'))", 'sympy_rhs': "Mul(exp(Mul(Integer(-1), Symbol('pdg0001575'), Symbol('pdg0004621'))), Abs(Symbol('pdg0004453')))", 'latex_lhs': 'A^*', 'latex_rhs': '|A| \\exp(-i \\\\theta)', 'latex_relation': '='}, '6556875579': {'sympy_lhs': "Mul(Pow(Symbol('pdg0002435'), Integer(-1)), Symbol('pdg0008251'))", 'sympy_rhs': 'Integer(2)', 'latex_lhs': '\\\\frac{I_{\\\\rm coherent}}{I_{\\\\rm incoherent}}', 'latex_rhs': '2', 'latex_relation': '='}, '6774684564': {'sympy_lhs': "Symbol('pdg0001575')", 'sympy_rhs': "Symbol('pdg0008586')", 'latex_lhs': '\\\\theta', 'latex_rhs': '\\phi', 'latex_relation': '='}, '7107090465': {'sympy_lhs': "Mul(Symbol('pdg0004698'), conjugate(Symbol('pdg0004698')))", 'sympy_rhs': "Pow(Abs(Symbol('pdg0004698')), Integer(2))", 'latex_lhs': 'B B^*', 'latex_rhs': '|B|^2', 'latex_relation': '='}, '7607271250': {'sympy': "Symbol('pdg0001575')", 'latex': '\\\\theta'}, '7621705408': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Add(Mul(exp(Mul(Symbol('pdg0001575'), Symbol('pdg0004621'))), exp(Mul(Integer(-1), Symbol('pdg0004621'), Symbol('pdg0008586'))), Abs(Mul(Symbol('pdg0004453'), Symbol('pdg0004698'), Abs(Add(Abs(Symbol('pdg0004453')), Mul(exp(Mul(Integer(-1), Symbol('pdg0001575'), Symbol('pdg0004621'))), exp(Mul(Symbol('pdg0004621'), Symbol('pdg0008586'))), Abs(Symbol('pdg0004698')))))))), Pow(Abs(Symbol('pdg0004453')), Integer(2)), Pow(Abs(Symbol('pdg0004698')), Integer(2)))", 'latex_lhs': 'I', 'latex_rhs': '|A|^2 + |B|^2 + |A| |B| \\exp(-i \\\\theta) \\exp(i \\phi) + |A| |B| \\exp(i \\\\theta) \\exp(-i \\phi)', 'latex_relation': '='}, '8046208134': {'sympy_lhs': "Symbol('pdg0008251')", 'sympy_rhs': "Mul(Integer(4), Pow(Abs(Symbol('pdg0004453')), Integer(2)))", 'latex_lhs': 'I_{\\\\rm coherent}', 'latex_rhs': '|A|^2 + |A|^2 + |A| |A| 2', 'latex_relation': '='}, '8065128065': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Add(Mul(Symbol('pdg0004453'), conjugate(Symbol('pdg0004453'))), Mul(Symbol('pdg0004453'), conjugate(Symbol('pdg0004698'))), Mul(Symbol('pdg0004698'), conjugate(Symbol('pdg0004453'))), Mul(Symbol('pdg0004698'), conjugate(Symbol('pdg0004698'))))", 'latex_lhs': 'I', 'latex_rhs': 'A A^* + B B^* + A B^* + B A^*', 'latex_relation': '='}, '8283354808': {'sympy_lhs': "Symbol('pdg0008251')", 'sympy_rhs': "Add(Pow(Abs(Symbol('pdg0004453')), Integer(2)), Mul(Integer(2), cos(Integer(0)), Abs(Symbol('pdg0004453')), Abs(Symbol('pdg0004698'))), Pow(Abs(Symbol('pdg0004698')), Integer(2)))", 'latex_lhs': 'I_{\\\\rm coherent}', 'latex_rhs': '|A|^2 + |B|^2 + |A| |B| 2 \\cos( 0 )', 'latex_relation': '='}, '8396997949': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Pow(Abs(Add(Symbol('pdg0004453'), Symbol('pdg0004698'))), Integer(2))", 'latex_lhs': 'I', 'latex_rhs': '| A + B |^2', 'latex_relation': '='}, '8497631728': {'sympy_lhs': "Symbol('pdg0007882')", 'sympy_rhs': "Add(Mul(Integer(2), cos(Add(Symbol('pdg0001575'), Mul(Integer(-1), Symbol('pdg0008586')))), Abs(Symbol('pdg0004453')), Abs(Symbol('pdg0004698'))), Pow(Abs(Symbol('pdg0004453')), Integer(2)), Pow(Abs(Symbol('pdg0004698')), Integer(2)))", 'latex_lhs': 'I', 'latex_rhs': '|A|^2 + |B|^2 + |A| |B| 2 \\cos( \\\\theta - \\phi )', 'latex_relation': '='}, '8602221482': {'sympy_lhs': "cos(Add(Symbol('pdg0001575'), Mul(Integer(-1), Symbol('pdg0008586'))))", 'sympy_rhs': 'Integer(0)', 'latex_lhs': '\\langle \\cos(\\\\theta - \\phi) \\\\rangle', 'latex_rhs': '0', 'latex_relation': '='}, '9191880568': {'sympy_lhs': "Symbol('pdg0003192')", 'sympy_rhs': '', 'latex_lhs': 'Z Z^*', 'latex_rhs': '|Z| |Z| \\exp( -i \\\\theta ) \\exp( i \\\\theta )', 'latex_relation': '='}}
EXPECTED_STEPS = [('111484', {'4182362050', '1928085940'}), ('111253', {'9191880568', '4182362050', '1928085940'}), ('111457', {'9191880568', '3350830826'}), ('111886', {'3350830826', '2236639474'}), ('111556', {'1020854560', '2236639474', '8396997949'}), ('111474', {'6306552185', '1020854560'}), ('111457', {'6306552185', '8065128065'}), ('111886', {'3350830826', '4075539836'}), ('111886', {'3350830826', '7107090465'}), ('111556', {'8065128065', '4075539836', '5125940051'}), ('111556', {'1525861537', '7107090465', '5125940051'}), ('111984', {'7607271250', '4192519596'}), ('111886', {'4182362050', '1357848476'}), ('111484', {'6555185548', '1357848476'}), ('111484', {'4504256452', '4192519596'}), ('111797', {'4504256452', '7621705408', '1357848476', '4192519596'}), ('111457', {'7621705408', '3085575328'}), ('111886', {'4585932229', '3660957533'}), ('111182', {'2700934933', '3660957533'}), ('111556', {'8497631728', '2700934933', '3085575328'}), ('111556', {'8283354808', '8497631728', '6774684564'}), ('111556', {'8283354808', '2719691582', '8046208134'}), ('111457', {'1172039918', '8046208134'}), ('111556', {'8497631728', '6240206408', '8602221482'}), ('111556', {'6529793063', '2719691582', '6240206408'}), ('111457', {'6529793063', '3060393541'}), ('111421', {'6556875579', '1172039918', '3060393541'})]
FEEDS = [[], [], [], ["Symbol('pdg0003192')", "Add(Symbol('pdg0004453'), Symbol('pdg0004698'))"], [], [], [], ["Symbol('pdg0003192')", "Symbol('pdg0004453')"], ["Symbol('pdg0004698')", "Symbol('pdg0003192')"], [], [], ["Symbol('pdg0003192')", "Symbol('pdg0004698')"], ["Symbol('pdg0004453')", "Symbol('pdg0003192')"], [], [], [], [], ["Symbol('pdg0004037')", "Add(Mul(Integer(-1), Symbol('pdg0008586')), Symbol('pdg0001575'))"], ['Integer(2)'], [], [], [], [], [], [], [], []]
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
        if len(nodes)!=27 or len(bindings)!=66 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=27,source_bindings=66,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Restore conjugate/product ASTs and use mathematical I for the imaginary unit, not an unconstrained symbol.',
                     'Repair nested absolute values in cross-term ASTs from the reviewed LaTex.',
                     'Step12 binds a theta feed instead of the polar-Z expression; restore the polar expression and both variable substitutions.',
                     'Step16 lacks the A-conjugate and expanded intensity bindings; explicitly supply all four polar factors and the intensity definition in reconstruction.',
                     'Correct reversed or incomplete variable feeds and replace x by theta-phi on BOTH sides of the cosine identity.',
                     'Restore the ensemble-average meaning of the incoherent cross term; its stored AST loses averaging.',
                     'Distinguish arbitrary coherent fixed phase from constructive phase alignment.',
                     'The factor-two ratio requires equal nonzero amplitudes; both amplitudes zero makes the ratio undefined.',
                     'Use common polarization/mode, common intensity normalization and fixed amplitudes; uniform relative phase is a sufficient incoherent averaging model.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_wave_interference_source.py','sciona/physics_ingest/wave_interference_proof.py']})

    historical=json.loads((root/'docs/reviews/wave_interference_execution.json').read_text())['source_proof']
    for name,digest in historical['implementation_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Historical source implementation changed')
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
