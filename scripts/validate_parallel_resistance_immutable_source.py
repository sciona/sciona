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
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars
from sciona.ghost.dimensions import DimensionalSignature
from sciona.physics_ingest.parallel_resistance_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1457415749': {'latex_lhs': '\\\\frac{1}{R_{\\\\rm total}}', 'latex_rhs': '\\\\frac{1}{R_1} + \\\\frac{1}{R_2}', 'latex_relation': '=', 'sympy_lhs': "Pow(Symbol('pdg0001908'), Integer(-1))", 'sympy_rhs': "Add(Pow(Symbol('pdg0008697'), Integer(-1)), Pow(Symbol('pdg0003461'), Integer(-1)))"}, '2051901211': {'latex_lhs': '\\\\frac{V}{R_1}', 'latex_rhs': 'I_1', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0006599'), Pow(Symbol('pdg0008697'), Integer(-1)))", 'sympy_rhs': "Symbol('pdg0003978')"}, '2271186630': {'latex_lhs': 'V', 'latex_rhs': 'I_{\\\\rm total} R_{\\\\rm total}', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0006599')", 'sympy_rhs': "Mul(Symbol('pdg0001908'), Symbol('pdg0009647'))"}, '2809345867': {'latex_lhs': '\\\\frac{V}{R_{\\\\rm total}}', 'latex_rhs': 'I_{\\\\rm total}', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('pdg0001908'), Integer(-1)), Symbol('pdg0006599'))", 'sympy_rhs': "Symbol('pdg0009647')"}, '4087145886': {'latex_lhs': 'V', 'latex_rhs': 'I R', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0006599')", 'sympy_rhs': "Mul(Symbol('pdg0004501'), Symbol('pdg0006458'))"}, '4128500715': {'latex_lhs': 'V', 'latex_rhs': 'I_1 R_1', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0006599')", 'sympy_rhs': "Mul(Symbol('pdg0003978'), Symbol('pdg0008697'))"}, '4866160902': {'latex_lhs': '\\\\frac{V}{R_{\\\\rm total}}', 'latex_rhs': '\\\\frac{V}{R_1} + \\\\frac{V}{R_2}', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('pdg0001908'), Integer(-1)), Symbol('pdg0006599'))", 'sympy_rhs': "Add(Mul(Symbol('pdg0006599'), Pow(Symbol('pdg0008697'), Integer(-1))), Mul(Pow(Symbol('pdg0003461'), Integer(-1)), Symbol('pdg0006599')))"}, '6753224061': {'latex_lhs': 'I_{\\\\rm total}', 'latex_rhs': 'I_1 + I_2', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0009647')", 'sympy_rhs': "Add(Symbol('pdg0003978'), Symbol('pdg0004856'))"}, '7002609475': {'latex_lhs': '\\\\frac{V}{R_2}', 'latex_rhs': 'I_2', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('pdg0003461'), Integer(-1)), Symbol('pdg0006599'))", 'sympy_rhs': "Symbol('pdg0004856')"}, '9243879541': {'latex_lhs': 'V', 'latex_rhs': 'I_2 R_2', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0006599')", 'sympy_rhs': "Mul(Symbol('pdg0003461'), Symbol('pdg0004856'))"}}
EXPECTED_STEPS=[
 ('111984',{'4087145886','4128500715'}),('111984',{'4087145886','9243879541'}),
 ('111975',{'9243879541','7002609475'}),('111975',{'4128500715','2051901211'}),
 ('111984',{'4087145886','2271186630'}),('111975',{'2271186630','2809345867'}),
 ('111246',{'2809345867','2051901211','7002609475','6753224061','4866160902'}),
 ('111975',{'4866160902','1457415749'})]
FEEDS=[['pdg0004501','pdg0003978','pdg0006458','pdg0008697'],
       ['pdg0004501','pdg0004856','pdg0006458','pdg0003461'],['pdg0003461'],['pdg0008697'],
       ['pdg0004501','pdg0009647','pdg0006458','pdg0001908'],['pdg0001908'],[],['pdg0006599']]
MISSING={'4087145886'}



def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.artifact_id::text,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if graph != dict(artifact_id='121cbf9d-f0e5-5833-a02e-af773a519f2a',content_hash=SOURCE_HASH):
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=8 or len(bindings)!=19 or any(b['status']!='active' for b in bindings):
            raise ValueError('Source inventory differs')
        for i,(node,(rule,identities)) in enumerate(zip(nodes,EXPECTED_STEPS)):
            sig=json.loads(node['type_signature'])
            actual={b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            feeds=[f['sympy'] for f in sig['variable_bindings']['feeds']]
            expected_feeds=["Symbol('"+v+"')" for v in FEEDS[i]]
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
        raise ValueError('Nine stored equations and one recovered source equation required')
    for name,path in [('symbols',symbol_file),('infrules',rule_file),('expr_and_feed',expression_file)]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pins[0]['conversion_of_data_formats/'+name+'.cypher']:
            raise ValueError('Public source file pin differs')
    rules=load_pinned_rule_contracts(rule_file.read_bytes(),pins[0]['conversion_of_data_formats/infrules.cypher'])
    if any(rule not in rules for rule,_ in EXPECTED_STEPS): raise ValueError('Missing rule contract')
    definitions=load_pinned_pdg_scalars(symbol_file.read_bytes(),pins[0]['conversion_of_data_formats/symbols.cypher'])
    current=DimensionalSignature(I=1);resistance=DimensionalSignature(M=1,L=2,T=-3,I=-2)
    voltage=DimensionalSignature(M=1,L=2,T=-3,I=-1)
    dimensions={**dict.fromkeys(['pdg0004501','pdg0003978','pdg0004856','pdg0009647'],current),
                **dict.fromkeys(['pdg0006458','pdg0008697','pdg0003461','pdg0001908'],resistance),
                'pdg0006599':voltage}
    if any(definitions[k].dimension!=v for k,v in dimensions.items()) or current.multiply(resistance)!=voltage:
        raise ValueError('Electrical dimensions differ')
    found=set()
    for block in re.split(r'(?m)^UNWIND ',expression_file.read_text()):
        match=re.match(r'\[\{id:"(\d+)"',block)
        if not match or match[1] not in EXPECTED: continue
        fields=dict(re.findall(r'(\w+):"((?:\\.|[^"\\])*)"',block))
        if match[1] in found or {k:fields.get(k) for k in EXPECTED[match[1]]}!=EXPECTED[match[1]]:
            raise ValueError('Public equation changed/duplicated')
        found.add(match[1])
    if found!=set(EXPECTED): raise ValueError('Public equation missing')
    return dict(approved=False,source_nodes=8,source_bindings=19,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        recovered_missing_equations=sorted(MISSING),electrical_dimensions_verified=9,
        corrections=['Recover Ohms-law equation from original pinned source; three missing binding occurrences reference this one identity.',
                     'State common two-node voltage, ideal positive linear resistors and consistent current orientation.',
                     'Resistor divisions require nonzero resistance; final source division requires nonzero voltage.',
                     'Extend to zero voltage by the constitutive model, not by dividing zero current into zero voltage.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_parallel_resistance_immutable_source.py','sciona/physics_ingest/parallel_resistance_proof.py']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['approved','source_nodes','source_bindings','corrections']}))
