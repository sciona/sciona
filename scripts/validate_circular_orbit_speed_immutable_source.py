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
from sciona.physics_ingest.circular_orbit_speed_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'3046191961': {'latex_lhs': 'v_{\\\\rm Earth\\ orbit}', 'latex_rhs': '\\\\frac{C_{\\\\rm Earth\\ orbit}}{t_{\\\\rm Earth\\ orbit}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0007427')", 'sympy_rhs': "Mul(Symbol('pdg0001534'), Pow(Symbol('pdg0005344'), Integer(-1)))"}, '3080027960': {'latex_lhs': 'v_{\\\\rm Earth\\ orbit}', 'latex_rhs': '\\\\frac{2 \\pi r_{\\\\rm Earth\\ orbit}}{t_{\\\\rm Earth\\ orbit}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0007427')", 'sympy_rhs': "Mul(Integer(2), Symbol('pdg0003141'), Pow(Symbol('pdg0005344'), Integer(-1)), Symbol('pdg0006081'))"}, '3472836147': {'latex_lhs': 'r_{\\\\rm Earth\\ orbit}', 'latex_rhs': '1.496\\ 10^8 {\\\\rm km}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0006081')", 'sympy_rhs': "Float('1.496', precision=53)"}, '4180845508': {'latex_lhs': 'v_{\\\\rm Earth\\ orbit}', 'latex_rhs': '29.8 \\\\frac{{\\\\rm km}}{{\\\\rm sec}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0007427')", 'sympy_rhs': "Float('29.800000000000001', precision=53)"}, '4593428198': {'latex_lhs': 'v_{\\\\rm Earth\\ orbit}', 'latex_rhs': '\\\\frac{2 \\pi r_{\\\\rm Earth\\ orbit}}{3.16\\ 10^7 {\\\\rm seconds}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0007427')", 'sympy_rhs': "Mul(Float('0.63291139240506322', precision=53), Symbol('pdg0003141'), Symbol('pdg0006081'))"}, '5426308937': {'latex_lhs': 'v', 'latex_rhs': '\\\\frac{d}{t}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0001943'))"}, '6348260313': {'latex_lhs': 'C_{\\\\rm Earth\\ orbit}', 'latex_rhs': '2 \\pi r_{\\\\rm Earth\\ orbit}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001534')", 'sympy_rhs': "Mul(Integer(2), Symbol('pdg0003141'), Symbol('pdg0006081'))"}, '6785303857': {'latex_lhs': 'C', 'latex_rhs': '2 \\pi r', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0003034')", 'sympy_rhs': "Mul(Integer(2), Symbol('pdg0002530'), Symbol('pdg0003141'))"}, '6946088325': {'latex_lhs': 'v', 'latex_rhs': '\\\\frac{C}{t}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0001357')", 'sympy_rhs': "Mul(Pow(Symbol('pdg0001467'), Integer(-1)), Symbol('pdg0003034'))"}, '6998364753': {'latex_lhs': 'v_{\\\\rm Earth\\ orbit}', 'latex_rhs': '\\\\frac{2 \\pi \\left( 1.496\\ 10^8 {\\\\rm km} \\\\right)}{3.16\\ 10^7 {\\\\rm seconds}}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0007427')", 'sympy_rhs': "Mul(Float('0.63291139240506322', precision=53), Symbol('pdg0003141'))"}, '7175416299': {'latex_lhs': 't_{\\\\rm Earth\\ orbit}', 'latex_rhs': '1 {\\\\rm year}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0005344')", 'sympy_rhs': 'Integer(1)'}, '8721295221': {'latex_lhs': 't_{\\\\rm Earth\\ orbit}', 'latex_rhs': '3.16 10^7 {\\\\rm seconds}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0005344')", 'sympy_rhs': 'Integer(3)'}}
EXPECTED_STEPS = [
 ('111886',{'5426308937','6946088325'}),('111984',{'6785303857','6348260313'}),
 ('111236',{'6946088325','3046191961'}),('111556',{'6348260313','3046191961','3080027960'}),
 ('111646',{'7175416299','8721295221'}),('111556',{'8721295221','3080027960','4593428198'}),
 ('111556',{'3472836147','4593428198','6998364753'}),('111457',{'6998364753','4180845508'})]
FEEDS=[['pdg0001943','pdg0003034'],['pdg0003034','pdg0001534','pdg0002530','pdg0006081'],
       ['pdg0003034','pdg0001534','pdg0001357','pdg0007427','pdg0001467','pdg0005344'],[],[],[],[],[]]
MISSING={'5426308937','6785303857'}



def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash']!=SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=8 or len(bindings)!=19 or any(b['status']!='active' for b in bindings):
            raise ValueError('Source inventory differs')
        for i,(node,(rule,identities)) in enumerate(zip(nodes,EXPECTED_STEPS)):
            sig=json.loads(node['type_signature'])
            actual={b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            feeds=[f['sympy'] for f in sig['variable_bindings']['feeds']]
            expected_feeds=["Symbol('"+v+"')" for v in FEEDS[i]] if i!=4 else ['Integer(365)']
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
    result=dict(approved=False,source_nodes=8,source_bindings=19,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Recover speed definition and circle circumference from pinned original source.',
                     'Interpret pi as exact constant; retain scientific notation powers and length/time units omitted from symbolic fields.',
                     'Unity conversion feed stores only365; full365*24*60*60 yields31536000 seconds.',
                     'Use exact unit conversion before final rounding; source rounded31600000seconds yields29.7457 rather than29.8061km/s.',
                     'Final29.8km/s is an approximation; reconstructed exact365day example rounds to it.',
                     'Circular-path average speed only; uniform circular motion required for instantaneous interpretation.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_circular_orbit_speed_source.py','sciona/physics_ingest/circular_orbit_speed_proof.py']})

    historical=json.loads((root/'docs/reviews/circular_orbit_speed_execution.json').read_text())['source_proof']
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
