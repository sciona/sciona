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
from sciona.physics_ingest.brewster_angle_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1310571337': {'latex_lhs': '\\\\theta_{\\\\rm refracted}', 'latex_rhs': '90^{\\circ} - \\\\theta_{\\\\rm Brewster}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004928')", 'sympy_rhs': ''}, '2575937347': {'latex_lhs': 'n_1 \\sin( \\\\theta_{\\\\rm Brewster} )', 'latex_rhs': 'n_2 \\sin( \\\\theta_{\\\\rm refracted} )', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0002941'), sin(Symbol('pdg0004928')))", 'sympy_rhs': "Mul(Symbol('pdg0001958'), sin(Symbol('pdg0002243')))"}, '2768857871': {'latex_lhs': '\\\\frac{\\sin( \\\\theta_{\\\\rm Brewster} )}{\\cos( \\\\theta_{\\\\rm Brewster} )}', 'latex_rhs': '\\\\frac{n_2}{n_1}', 'latex_relation': '=', 'sympy_lhs': "Mul(sin(Symbol('pdg0004928')), Pow(cos(Symbol('pdg0004928')), Integer(-1)))", 'sympy_rhs': "Mul(Symbol('pdg0001958'), Pow(Symbol('pdg0002941'), Integer(-1)))"}, '3061811650': {'latex_lhs': 'n_1 \\sin( \\\\theta_{\\\\rm Brewster} )', 'latex_rhs': 'n_2 \\cos( \\\\theta_{\\\\rm Brewster} )', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0002941'), sin(Symbol('pdg0004928')))", 'sympy_rhs': "Mul(Symbol('pdg0001958'), cos(Symbol('pdg0004928')))"}, '3417126140': {'latex_lhs': '\\\\tan( \\\\theta_{\\\\rm Brewster} )', 'latex_rhs': '\\\\frac{ n_2 }{ n_1 }', 'latex_relation': '=', 'sympy_lhs': "tan(Symbol('pdg0004928'))", 'sympy_rhs': "Mul(Symbol('pdg0001958'), Pow(Symbol('pdg0002941'), Integer(-1)))"}, '4501377629': {'latex_lhs': '\\\\tan( \\\\theta_{\\\\rm Brewster} )', 'latex_rhs': '\\\\frac{ \\sin( \\\\theta_{\\\\rm Brewster} )}{\\cos( \\\\theta_{\\\\rm Brewster} )}', 'latex_relation': '=', 'sympy_lhs': "tan(Symbol('pdg0004928'))", 'sympy_rhs': "Mul(sin(Symbol('pdg0004928')), Pow(cos(Symbol('pdg0004928')), Integer(-1)))"}, '4968680693': {'latex_lhs': '\\\\tan( x )', 'latex_rhs': '\\\\frac{ \\sin( x )}{\\cos( x )}', 'latex_relation': '=', 'sympy_lhs': "tan(Symbol('pdg0001464'))", 'sympy_rhs': "Mul(sin(Symbol('pdg0001464')), Pow(cos(Symbol('pdg0001464')), Integer(-1)))"}, '6450985774': {'latex_lhs': 'n_1 \\sin( \\\\theta_1 )', 'latex_rhs': 'n_2 \\sin( \\\\theta_2 )', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0002941'), sin(Symbol('pdg0003509')))", 'sympy_rhs': "Mul(Symbol('pdg0001958'), sin(Symbol('pdg0007545')))"}, '6831637424': {'latex_lhs': '\\sin( 90^{\\circ} - \\\\theta_{\\\\rm Brewster} )', 'latex_rhs': '\\cos( \\\\theta_{\\\\rm Brewster} )', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004928')", 'sympy_rhs': ''}, '7696214507': {'latex_lhs': 'n_1 \\sin( \\\\theta_{\\\\rm Brewster} )', 'latex_rhs': 'n_2 \\sin( 90^{\\circ} - \\\\theta_{\\\\rm Brewster} )', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0002941'), sin(Symbol('pdg0004928')))", 'sympy_rhs': "Mul(Symbol('pdg0001958'), sin(Add(Integer(90), Mul(Integer(-1), Symbol('pdg0004928')))))"}, '8495187962': {'latex_lhs': '\\\\theta_{\\\\rm Brewster}', 'latex_rhs': '\\arctan{ \\left( \\\\frac{ n_1 }{ n_2 } \\\\right) }', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004928')", 'sympy_rhs': "atan(Mul(Pow(Symbol('pdg0001958'), Integer(-1)), Symbol('pdg0002941')))"}, '8588429722': {'latex_lhs': '\\sin( 90^{\\circ} - x )', 'latex_rhs': '\\cos( x )', 'latex_relation': '=', 'sympy_lhs': "sin(Add(Integer(90), Mul(Integer(-1), Symbol('pdg0001464'))))", 'sympy_rhs': "cos(Symbol('pdg0001464'))"}, '8945218208': {'latex_lhs': '\\\\theta_{\\\\rm Brewster} + \\\\theta_{\\\\rm refracted}', 'latex_rhs': '90^{\\circ}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004928')", 'sympy_rhs': ''}, '9756089533': {'latex_lhs': '\\sin( \\\\theta_{\\\\rm Brewster} )', 'latex_rhs': '\\\\frac{n_2}{n_1} \\cos( \\\\theta_{\\\\rm Brewster} )', 'latex_relation': '=', 'sympy_lhs': "sin(Symbol('pdg0004928'))", 'sympy_rhs': "Mul(Symbol('pdg0001958'), Pow(Symbol('pdg0002941'), Integer(-1)), cos(Symbol('pdg0004928')))"}}
EXPECTED_STEPS=[
 ('111282',{'8945218208','1310571337'}),('111984',{'6450985774','2575937347'}),
 ('111556',{'1310571337','2575937347','7696214507'}),('111886',{'8588429722','6831637424'}),
 ('111556',{'6831637424','7696214507','3061811650'}),('111975',{'3061811650','9756089533'}),
 ('111975',{'9756089533','2768857871'}),('111886',{'4968680693','4501377629'}),
 ('111556',{'4501377629','2768857871','3417126140'}),('111490',{'3417126140','8495187962'})]
FEEDS=[
 ["Symbol('pdg0004928')"],
 ["Symbol('pdg0007545')","Symbol('pdg0002243')","Symbol('pdg0003509')","Symbol('pdg0004928')"],[],
 ["Symbol('pdg0004928')","Symbol('pdg0001464')"],[],["Symbol('pdg0002941')"],
 ["cos(Symbol('pdg0004928'))"],["Symbol('pdg0004928')","Symbol('pdg0001464')"],[],
 ["atan(Symbol('pdg0001464'))","Symbol('pdg0001464')"]]
MISSING=set()



def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash']!=SOURCE_HASH:
            raise ValueError('Immutable original source hash differs')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY length(node_id),node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=10 or len(bindings)!=23 or any(b['status']!='active' for b in bindings):
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
        raise ValueError('Fourteen stored source equations required')
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
    result=dict(approved=False,source_nodes=10,source_bindings=23,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),
        corrections=['Final source arctan reverses ratio; use n2/n1 with n1 incident medium.',
                     'Convert90degrees to pi/2 radians in trigonometric ASTs.',
                     'Reconstruct missing angle sums/complements/cofunction expressions from LaTeX.',
                     'Reverse source renaming feeds for cofunction and tangent identities: generic x becomes Brewster angle.',
                     'Require positive indices and acute physical branch for inverse tangent and cosine division.',
                     'Equal-index result is complementary-angle convention, not unique reflection zero.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_brewster_angle_source.py','sciona/physics_ingest/brewster_angle_proof.py']})

    historical=json.loads((root/'docs/reviews/brewster_angle_execution.json').read_text())['source_proof']
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
