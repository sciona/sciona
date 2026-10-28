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
from sciona.physics_ingest.gravity_mass_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1132941271': {'latex_lhs': 'm_{\\\\rm Earth}', 'latex_rhs': '\\\\frac{(9.80665 m/s^2) (6.3781*10^6 m)^2}{6.67430*10^{-11}m^3 kg^{-1} s^{-2}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0005458')", 'sympy_rhs': "Float('6.3780999999999999', precision=53)"}, '2308660627': {'latex_lhs': 'G \\\\frac{m_{\\\\rm Earth}}{r_{\\\\rm Earth}^2}', 'latex_rhs': 'g_{\\\\rm Earth}', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('pdg0003236'), Integer(-2)), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'sympy_rhs': "Symbol('pdg0007557')"}, '2484824786': {'latex_lhs': 'F', 'latex_rhs': 'm g', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Symbol('pdg0001649'), Symbol('pdg0005156'))"}, '3364286646': {'latex_lhs': 'm_{\\\\rm Earth}', 'latex_rhs': '5.972*10^{24} kg', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0005458')", 'sympy_rhs': "Mul(Float('5.9720000000000003e+24', precision=53), Symbol('kg'))"}, '4800170179': {'latex_lhs': 'F', 'latex_rhs': 'm g_{\\\\rm Earth}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Symbol('pdg0005156'), Symbol('pdg0007557'))"}, '5345738321': {'latex_lhs': 'F', 'latex_rhs': 'm a', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Symbol('pdg0005156'), Symbol('pdg0009140'))"}, '6935745841': {'latex_lhs': 'F', 'latex_rhs': 'G \\\\frac{m_1 m_2}{x^2}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0004037'), Integer(-2)), Symbol('pdg0004851'), Symbol('pdg0005022'), Symbol('pdg0006277'))"}, '7112613117': {'latex_lhs': 'm_{\\\\rm Earth}', 'latex_rhs': '\\\\frac{(9.80665 m/s^2) r_{\\\\rm Earth}^2}{6.67430*10^{-11}m^3 kg^{-1} s^{-2}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0005458')", 'sympy_rhs': ''}, '7846240076': {'latex_lhs': 'm_{\\\\rm Earth}', 'latex_rhs': '\\\\frac{(9.80665 m/s^2) r_{\\\\rm Earth}^2}{G}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0005458')", 'sympy_rhs': "Mul(Integer(9), Pow(Symbol('pdg0003236'), Integer(2)), Pow(Symbol('pdg0006277'), Integer(-1)))"}, '8661803554': {'latex_lhs': 'F', 'latex_rhs': 'G \\\\frac{m_{\\\\rm Earth} m}{r_{\\\\rm Earth}^2}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004202')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0003236'), Integer(-2)), Symbol('pdg0005156'), Symbol('pdg0005458'), Symbol('pdg0006277'))"}, '9407192813': {'latex_lhs': 'G \\\\frac{m_{\\\\rm Earth} m}{r_{\\\\rm Earth}^2}', 'latex_rhs': 'm g_{\\\\rm Earth}', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('pdg0003236'), Integer(-2)), Symbol('pdg0005156'), Symbol('pdg0005458'), Symbol('pdg0006277'))", 'sympy_rhs': "Mul(Symbol('pdg0005156'), Symbol('pdg0007557'))"}, '9440616166': {'latex_lhs': 'm_{\\\\rm Earth}', 'latex_rhs': '\\\\frac{g_{\\\\rm Earth} r_{\\\\rm Earth}^2}{G}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0005458')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0003236'), Integer(2)), Pow(Symbol('pdg0006277'), Integer(-1)), Symbol('pdg0007557'))"}}
EXPECTED_STEPS=[
 ('111886',{'5345738321','2484824786'}),('111886',{'2484824786','4800170179'}),
 ('111236',{'6935745841','8661803554'}),('111355',{'8661803554','4800170179','9407192813'}),
 ('111975',{'9407192813','2308660627'}),('111182',{'2308660627','9440616166'}),
 ('111715',{'9440616166','7846240076'}),('111715',{'7846240076','7112613117'}),
 ('111715',{'7112613117','1132941271'}),('111457',{'1132941271','3364286646'})]
FEEDS=[
 ["Symbol('pdg0009140')","Symbol('pdg0001649')"],
 ["Symbol('pdg0001649')","Symbol('pdg0007557')"],
 ["Symbol('pdg0005458')","Symbol('pdg0005458')","Symbol('pdg0004851')","Symbol('pdg0005156')","Symbol('pdg0004037')","Symbol('pdg0003236')"],
 [],["Symbol('pdg0005156')"],
 ["Mul(Pow(Symbol('pdg0006277'), Integer(-1)), Pow(Symbol('pdg0003236'), Integer(2)))"],
 ["Symbol('pdg0007557')","Float('9.8066499999999994', precision=53)",'Pow(Mul(Pow(second, Integer(-1)),meter), Integer(2))'],
 ["Symbol('pdg0006277')","Mul(Float('6.6742999999999997', precision=53), Pow(Integer(10), Mul(Integer(-1), Integer(11))))",'Mul(Pow(meter, Integer(3)), Pow(second, Integer(-2)))'],
 ["Symbol('pdg0003236')","Mul(Float('6.3780999999999999', precision=53), Pow(Integer(10), Integer(6)))","Symbol('pdg0005156')"],[]]
MISSING={'5345738321','6935745841'}



def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=10 or len(bindings)!=21 or any(b['status']!='active' for b in bindings):
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
        raise ValueError('Ten stored equations and two recovered source equations required')
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
    result=dict(approved=False,source_nodes=10,source_bindings=21,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Recover Newton second law and inverse-square gravitation from pinned source.',
                     'Rename m1 identity pdg0005022 to source mass pdg0005458; source feed incorrectly repeats target identity.',
                     'Interpret acceleration units as m/s², G units as m³/kg/s² and radius units as meters, not mass symbol.',
                     'Reconstruct truncated/empty numerical ASTs from displayed source constants.',
                     'Source displayed inputs yield5.9771974175e24kg, not displayed5.972e24kg.',
                     'Require purely gravitational acceleration at known spherical-source radius; no effective-gravity/rotation correction implied.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_gravity_mass_source.py','sciona/physics_ingest/gravity_mass_proof.py']})


    historical=json.loads((root/'docs/reviews/gravity_mass_execution.json').read_text())['source_proof']
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
