"""Recheck immutable mechanical-energy source independently of current approval state."""
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
from sciona.physics_ingest.mechanical_energy_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1772416655': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0004550'), Mul(Integer(-1), Symbol('pdg0005579'))))", 'sympy_rhs': 'Integer(0)', 'latex_lhs': '\\\\frac{E_2 - E_1}{t}', 'latex_rhs': 'v F - F v', 'latex_relation': '='}, '1809909100': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0004550'), Mul(Integer(-1), Symbol('pdg0005579'))))", 'sympy_rhs': 'Integer(0)', 'latex_lhs': '\\\\frac{E_2 - E_1}{t}', 'latex_rhs': '0', 'latex_relation': '='}, '2186083170': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955'))))", 'sympy_rhs': "Mul(Symbol('pdg0001357'), Symbol('pdg0004202'))", 'latex_lhs': '\\\\frac{KE_2 - KE_1}{t}', 'latex_rhs': 'v F', 'latex_relation': '='}, '2431507955': {'sympy_lhs': "Symbol('pdg0008849')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0004202'), Symbol('pdg0005467'))", 'latex_lhs': 'PE_2', 'latex_rhs': '-F x_2', 'latex_relation': '='}, '2770069250': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0004550'), Mul(Integer(-1), Symbol('pdg0005579'))))", 'sympy_rhs': "Add(Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955')))), Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Mul(Integer(-1), Symbol('pdg0004093')), Symbol('pdg0008849'))))", 'latex_lhs': '\\\\frac{E_2 - E_1}{t}', 'latex_rhs': '\\\\frac{(KE_2 - KE_1)}{t} + \\\\frac{(PE_2 - PE_1)}{t}', 'latex_relation': '='}, '2857430695': {'sympy_lhs': "Symbol('pdg0009140')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Mul(Integer(-1), Symbol('pdg0002473')), Symbol('pdg0004770')))", 'latex_lhs': 'a', 'latex_rhs': '\\\\frac{v_2 - v_1}{t}', 'latex_relation': '='}, '3591237106': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0004550'), Mul(Integer(-1), Symbol('pg5579'))))", 'sympy_rhs': "Add(Mul(Integer(-1), Symbol('pdg0001357'), Symbol('pdg0004202')), Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955')))))", 'latex_lhs': '\\\\frac{E_2 - E_1}{t}', 'latex_rhs': '\\\\frac{(KE_2 - KE_1)}{t} - F v', 'latex_relation': '='}, '3806977900': {'sympy_lhs': "Add(Symbol('pdg0004550'), Mul(Integer(-1), Symbol('pdg0005579')))", 'sympy_rhs': 'Integer(0)', 'latex_lhs': 'E_2 - E_1', 'latex_rhs': '0', 'latex_relation': '='}, '4270680309': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955'))))", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0005156'), Add(Mul(Integer(-1), Pow(Symbol('pdg0002473'), Integer(2))), Pow(Symbol('pdg0004770'), Integer(2))))", 'latex_lhs': '\\\\frac{KE_2 - KE_1}{t}', 'latex_rhs': '\\\\frac{1}{2} m \\\\frac{\\left( v_2^2 - v_1^2 \\\\right)}{t}', 'latex_relation': '='}, '4303372136': {'sympy_lhs': "Symbol('pdg0005579')", 'sympy_rhs': "Add(Symbol('pdg0001955'), Symbol('pdg0004093'))", 'latex_lhs': 'E_1', 'latex_rhs': 'KE_1 + PE_1', 'latex_relation': '='}, '4648451961': {'sympy_lhs': "Add(Mul(Integer(-1), Pow(Symbol('pdg0002473'), Integer(2))), Pow(Symbol('pdg0004770'), Integer(2)))", 'sympy_rhs': "Mul(Add(Mul(Integer(-1), Symbol('pdg0002473')), Symbol('pdg0004770')), Add(Symbol('pdg0002473'), Symbol('pdg0004770')))", 'latex_lhs': 'v_2^2 - v_1^2', 'latex_rhs': '(v_2 + v_1)(v_2 - v_1)', 'latex_relation': '='}, '4669290568': {'sympy_lhs': "Symbol('pdg0004093')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0003852'), Symbol('pdg0004202'))", 'latex_lhs': 'PE_1', 'latex_rhs': '-F x_1', 'latex_relation': '='}, '4784793837': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955'))))", 'sympy_rhs': "Mul(Symbol('pdg0001357'), Symbol('pdg0005156'), Symbol('pdg0009140'))", 'latex_lhs': '\\\\frac{KE_2 - KE_1}{t}', 'latex_rhs': 'm v a', 'latex_relation': '='}, '4872970974': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Mul(Integer(-1), Symbol('pdg0004093')), Symbol('pdg0008849')))", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0001357'), Symbol('pdg0004202'))", 'latex_lhs': '\\\\frac{PE_2 - PE_1}{t}', 'latex_rhs': '-F v', 'latex_relation': '='}, '4928007622': {'sympy_lhs': "Symbol('pdg0001955')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0002473'), Integer(2)), Symbol('pdg0005156'))", 'latex_lhs': 'KE_1', 'latex_rhs': '\\\\frac{1}{2} m v_1^2', 'latex_relation': '='}, '5136652623': {'sympy_lhs': "Symbol('pdg0004931')", 'sympy_rhs': "Add(Symbol('pdg0004929'), Symbol('pdg0004930'))", 'latex_lhs': 'E', 'latex_rhs': 'KE + PE', 'latex_relation': '='}, '5345738321': {'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Symbol('pdg0005156'), Symbol('pdg0009140'))", 'latex_lhs': 'F', 'latex_rhs': 'm a', 'latex_relation': '='}, '5514556106': {'sympy_lhs': "Add(Symbol('pdg0004550'), Mul(Integer(-1), Symbol('pdg0005579')))", 'sympy_rhs': "Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955')), Mul(Integer(-1), Symbol('pdg0004093')), Symbol('pdg0008849'))", 'latex_lhs': 'E_2 - E_1', 'latex_rhs': '(KE_2 - KE_1) + (PE_2 - PE_1)', 'latex_relation': '='}, '5733146966': {'sympy_lhs': "Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955')))", 'sympy_rhs': "Mul(Rational(1, 2), Symbol('pdg0005156'), Add(Mul(Integer(-1), Pow(Symbol('pdg0002473'), Integer(2))), Pow(Symbol('pdg0004770'), Integer(2))))", 'latex_lhs': 'KE_2 - KE_1', 'latex_rhs': '\\\\frac{1}{2} m \\left(v_2^2 - v_1^2\\\\right)', 'latex_relation': '='}, '5781981178': {'sympy_lhs': "Add(Mul(Integer(-1), Pow(Symbol('pdg0001452'), Integer(2))), Pow(Symbol('pdg0001464'), Integer(2)))", 'sympy_rhs': "Mul(Add(Mul(Integer(-1), Symbol('pdg0001452')), Symbol('pdg0001464')), Add(Symbol('pdg0001452'), Symbol('pdg0001464')))", 'latex_lhs': 'x^2 - y^2', 'latex_rhs': '(x+y)(x-y)', 'latex_relation': '='}, '6715248283': {'sympy_lhs': "Symbol('pdg0004930')", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0004037'), Symbol('pdg0004202'))", 'latex_lhs': 'PE', 'latex_rhs': '-F x', 'latex_relation': '='}, '7267155233': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Mul(Integer(-1), Symbol('pdg0004093')), Symbol('pdg0008849')))", 'sympy_rhs': "Mul(Integer(-1), Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0004202'), Add(Mul(Integer(-1), Symbol('pdg0003852')), Symbol('pdg0005467')))", 'latex_lhs': '\\\\frac{PE_2 - PE_1}{t}', 'latex_rhs': '-F \\left( \\\\frac{x_2 - x_1}{t} \\\\right)', 'latex_relation': '='}, '7676652285': {'sympy_lhs': "Symbol('pdg0001352')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0004770'), Integer(2)), Symbol('pdg0005156'))", 'latex_lhs': 'KE_2', 'latex_rhs': '\\\\frac{1}{2} m v_2^2', 'latex_relation': '='}, '7734996511': {'sympy_lhs': "Add(Mul(Integer(-1), Symbol('pdg0004093')), Symbol('pdg0008849'))", 'sympy_rhs': "Mul(Integer(-1), Symbol('pdg0004202'), Add(Mul(Integer(-1), Symbol('pdg0003852')), Symbol('pdg0005467')))", 'latex_lhs': 'PE_2 - PE_1', 'latex_rhs': '-F ( x_2 - x_1 )', 'latex_relation': '='}, '7735737409': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955'))))", 'sympy_rhs': "Mul(Symbol('pdg0001357'), Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0005156'), Add(Mul(Integer(-1), Symbol('pdg0002473')), Symbol('pdg0004770')))", 'latex_lhs': '\\\\frac{KE_2 - KE_1}{t}', 'latex_rhs': 'm v \\\\frac{ v_2 - v_1 }{t}', 'latex_relation': '='}, '7875206161': {'sympy_lhs': "Symbol('pdg0004550')", 'sympy_rhs': "Add(Symbol('pdg0001352'), Symbol('pdg0008849'))", 'latex_lhs': 'E_2', 'latex_rhs': 'KE_2 + PE_2', 'latex_relation': '='}, '8357234146': {'sympy_lhs': "Symbol('pdg0004929')", 'sympy_rhs': "Mul(Rational(1, 2), Pow(Symbol('pdg0001357'), Integer(2)), Symbol('pdg0005156'))", 'latex_lhs': 'KE', 'latex_rhs': '\\\\frac{1}{2} m v^2', 'latex_relation': '='}, '8558338742': {'sympy_lhs': "Symbol('pdg0004550')", 'sympy_rhs': "Symbol('pdg0005579')", 'latex_lhs': 'E_2', 'latex_rhs': 'E_1', 'latex_relation': '='}, '9337785146': {'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Mul(Integer(-1), Symbol('pdg0003852')), Symbol('pdg0005467')))", 'latex_lhs': 'v', 'latex_rhs': '\\\\frac{x_2 - x_1}{t}', 'latex_relation': '='}, '9356924046': {'sympy_lhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Add(Symbol('pdg0001352'), Mul(Integer(-1), Symbol('pdg0001955'))))", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0005156'), Add(Mul(Integer(-1), Symbol('pdg0002473')), Symbol('pdg0004770')), Add(Mul(Rational(1, 2), Symbol('pdg0002473')), Mul(Rational(1, 2), Symbol('pdg0004770'))))", 'latex_lhs': '\\\\frac{KE_2 - KE_1}{t}', 'latex_rhs': 'm \\\\frac{v_2 + v_1}{2} \\\\frac{ v_2 - v_1 }{t}', 'latex_relation': '='}, '9397152918': {'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': "Add(Mul(Rational(1, 2), Symbol('pdg0002473')), Mul(Rational(1, 2), Symbol('pdg0004770')))", 'latex_lhs': 'v', 'latex_rhs': '\\\\frac{v_1 + v_2}{2}', 'latex_relation': '='}}
EXPECTED_STEPS = [('111236', {'5136652623', '7875206161'}), ('111236', {'5136652623', '4303372136'}), ('111222', {'7875206161', '4303372136', '5514556106'}), ('111984', {'8357234146', '7676652285'}), ('111984', {'8357234146', '4928007622'}), ('111222', {'7676652285', '5733146966', '4928007622'}), ('111975', {'5733146966', '4270680309'}), ('111984', {'5781981178', '4648451961'}), ('111634', {'9356924046', '4648451961', '4270680309'}), ('111634', {'9397152918', '9356924046', '7735737409'}), ('111634', {'4784793837', '2857430695', '7735737409'}), ('111634', {'5345738321', '2186083170', '4784793837'}), ('111984', {'2431507955', '6715248283'}), ('111984', {'6715248283', '4669290568'}), ('111222', {'2431507955', '7734996511', '4669290568'}), ('111975', {'7267155233', '7734996511'}), ('111634', {'4872970974', '7267155233', '9337785146'}), ('111975', {'2770069250', '5514556106'}), ('111634', {'4872970974', '2770069250', '3591237106'}), ('111634', {'2186083170', '1772416655', '3591237106'}), ('111457', {'1809909100', '1772416655'}), ('111182', {'1809909100', '3806977900'}), ('111530', {'8558338742', '3806977900'})]
FEEDS = [["Symbol('pdg0004931')", "Symbol('pdg0004550')", "Symbol('pdg0004929')", "Symbol('pdg0001352')", "Symbol('pdg0004930')", "Symbol('pdg0008849')"], ["Symbol('pdg0004931')", "Symbol('pdg0005579')", "Symbol('pdg0004929')", "Symbol('pdg0001955')", "Symbol('pdg0004930')", "Symbol('pdg0004093')"], [], ["Symbol('pdg0004929')", "Symbol('pdg0001352')", "Symbol('pdg0001357')", "Symbol('pdg0004770')"], ["Symbol('pdg0004929')", "Symbol('pdg0001955')", "Symbol('pdg0001357')", "Symbol('pdg0002473')"], [], ["Symbol('pdg0001467')"], ["Symbol('pdg0001452')", "Symbol('pdg0002473')", "Symbol('pdg0001464')", "Symbol('pdg0004770')"], [], [], [], [], ["Symbol('pdg0004930')", "Symbol('pdg0008849')", "Symbol('pdg0004037')", "Symbol('pdg0005467')"], ["Symbol('pdg0004930')", "Symbol('pdg0004093')", "Symbol('pdg0004037')", "Symbol('pdg0003852')"], [], ["Symbol('pdg0001467')"], [], ["Symbol('pdg0001467')"], [], [], [], ["Symbol('pdg0001467')"], ["Symbol('pdg0005579')"]]
MISSING = {'7676652285', '5345738321', '8357234146', '4928007622'}


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=23 or len(bindings)!=56 or any(b['status']!='active' for b in bindings):
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
    result=dict(approved=False,source_nodes=23,source_bindings=56,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Recover four missing public equations (kinetic energy and Newton law) from hash-pinned source.',
                     'Correct malformed pg5579 identifier to the initial total energy indicated by LaTex.',
                     'Restore the unsimplified v*F-F*v at step20; its stored AST is already zero.',
                     'Distinguish instantaneous endpoint velocities in kinetic energy from the interval-average velocity in finite-interval power equations.',
                     'Require constant mass, constant signed net force and a time-independent linear potential U=-F*x with no other work.',
                     'Endpoint mean equals interval average because acceleration F/m is constant; it is not a general identity.',
                     'Quotient steps require positive duration; establish duration zero separately with polynomial trajectories.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_mechanical_energy_source.py','sciona/physics_ingest/mechanical_energy_proof.py']})

    historical=json.loads((root/'docs/reviews/mechanical_energy_execution.json').read_text())['source_proof']
    for name,digest in historical['implementation_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Historical source implementation changed')
    if result!=historical:
        raise ValueError('Immutable source qualification differs')
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
