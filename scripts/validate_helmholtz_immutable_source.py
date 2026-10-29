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
from sciona.physics_ingest.helmholtz_proof import SOURCE_VERSION, SOURCE_HASH, build_proof, verify_proof

EXPECTED = {'1858578388': {'latex_lhs': '\\\\nabla^2 E( \\vec{r})\\exp(i \\omega t)', 'latex_rhs': '- \\omega^2 \\mu_0 \\epsilon_0 E( \\vec{r})\\exp(i \\omega t)', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('nabla'), Integer(2)), Function('pdg0006238')(Symbol('pdg0009472')), exp(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621'))))", 'sympy_rhs': "Mul(Integer(-1), Pow(Symbol('pdg0002321'), Integer(2)), Symbol('pdg0006197'), Symbol('pdg0007940'), Function('pdg0006238')(Symbol('pdg0009472')), exp(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621'))))"}, '2029293929': {'latex_lhs': '\\\\nabla^2 E( \\vec{r})\\exp(i \\omega t)', 'latex_rhs': '\\mu_0 \\epsilon_0 \\\\frac{\\partial^2}{\\partial t^2} E( \\vec{r})\\exp(i \\omega t)', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('nabla'), Integer(2)), Function('pdg0006238')(Symbol('pdg0009472')), exp(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621'))))", 'sympy_rhs': "Mul(Symbol('partial'), Pow(Symbol('pdg0001467'), Integer(-2)), Symbol('pdg0006197'), Symbol('pdg0007940'), Function('pdg0006238')(Symbol('pdg0009472')), exp(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621'))))"}, '3485475729': {'latex_lhs': '\\\\nabla^2 E( \\vec{r})', 'latex_rhs': '- \\\\frac{\\omega^2}{c^2} E( \\vec{r})', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('nabla'), Integer(2)), Function('pdg0006238')(Symbol('pdg0009472')))", 'sympy_rhs': "Mul(Integer(-1), Pow(Symbol('pdg0002321'), Integer(2)), Pow(Symbol('pdg0004567'), Integer(-2)), Function('pdg0006238')(Symbol('pdg0009472')))"}, '4585828572': {'latex_lhs': '\\epsilon_0 \\mu_0', 'latex_rhs': '\\\\frac{1}{c^2}', 'latex_relation': '=', 'sympy_lhs': "Mul(Symbol('pdg0006197'), Symbol('pdg0007940'))", 'sympy_rhs': "Pow(Symbol('pdg0004567'), Integer(-2))"}, '4985825552': {'latex_lhs': '\\\\nabla^2 E( \\vec{r})\\exp(i \\omega t)', 'latex_rhs': 'i \\omega \\mu_0 \\epsilon_0 \\\\frac{\\partial}{\\partial t} E( \\vec{r})\\exp(i \\omega t)', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('nabla'), Integer(2)), Function('pdg0006238')(Symbol('pdg0009472')), exp(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621'))))", 'sympy_rhs': "Mul(Symbol('pdg0002321'), Symbol('pdg0004621'), Symbol('pdg0006197'), Symbol('pdg0007940'), Derivative(Mul(Function('pdg0006238')(Symbol('pdg0009472')), exp(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621')))), Tuple(Symbol('pdg0001467'), Integer(1))))"}, '8494839423': {'latex_lhs': '\\\\nabla^2 \\vec{E}', 'latex_rhs': '\\mu_0 \\epsilon_0 \\\\frac{\\partial^2 \\vec{E}}{\\partial t^2}', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('nabla'), Integer(2)), Symbol('pdg0004326'))", 'sympy_rhs': "Mul(Symbol('partial'), Pow(Symbol('pdg0001467'), Integer(-2)), Symbol('pdg0004326'), Symbol('pdg0006197'), Symbol('pdg0007940'))"}, '8572852424': {'latex_lhs': '\\vec{E}', 'latex_rhs': 'E( \\vec{r},t)', 'latex_relation': '=', 'sympy_lhs': "Symbol('pdg0004326')", 'sympy_rhs': "Function('pdg0006238')(Symbol('pdg0009472'), Symbol('pdg0001467'))"}, '9394939493': {'latex_lhs': '\\\\nabla^2 E( \\vec{r},t)', 'latex_rhs': '\\mu_0 \\epsilon_0 \\\\frac{\\partial^2}{\\partial t^2} E( \\vec{r},t)', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('nabla'), Integer(2)), Function('pdg0006238')(Symbol('pdg0009472'), Symbol('pdg0001467')))", 'sympy_rhs': "Mul(Symbol('partial'), Pow(Symbol('pdg0001467'), Integer(-2)), Symbol('pdg0006197'), Symbol('pdg0007940'), Function('pdg0006238')(Symbol('pdg0009472'), Symbol('pdg0001467')))"}, '9485384858': {'latex_lhs': '\\\\nabla^2 E( \\vec{r})\\exp(i \\omega t)', 'latex_rhs': '- \\\\frac{\\omega^2}{c^2} E( \\vec{r})\\exp(i \\omega t)', 'latex_relation': '=', 'sympy_lhs': "Mul(Pow(Symbol('nabla'), Integer(2)), Function('pdg0002718')(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621'))), Function('pdg0006238')(Symbol('pdg0009472')))", 'sympy_rhs': "Mul(Integer(-1), Pow(Symbol('pdg0002321'), Integer(2)), Pow(Symbol('pdg0004567'), Integer(-2)), Function('pdg0002718')(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621'))), Function('pdg0006238')(Symbol('pdg0009472')))"}, '9499428242': {'latex_lhs': 'E( \\vec{r},t)', 'latex_rhs': 'E( \\vec{r})\\exp(i \\omega t)', 'latex_relation': '=', 'sympy_lhs': "Function('pdg0006238')(Symbol('pdg0009472'), Symbol('pdg0001467'))", 'sympy_rhs': "Mul(Function('pdg0002718')(Mul(Symbol('pdg0001467'), Symbol('pdg0002321'), Symbol('pdg0004621'))), Function('pdg0006238')(Symbol('pdg0009472')))"}}
EXPECTED_STEPS = [
 ('111237',{'8494839423','9499428242'}),
 ('111556',{'8572852424','8494839423','9394939493'}),
 ('111556',{'9499428242','9394939493','2029293929'}),
 ('111649',{'2029293929','4985825552'}),
 ('111649',{'4985825552','1858578388'}),
 ('111556',{'1858578388','4585828572','9485384858'}),
 ('111457',{'9485384858','3485475729'}),
]


def validate(root, symbol_file, rule_file, expression_file):
    records, pins = {}, []
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not graph or graph['content_hash'] != SOURCE_HASH:
            raise ValueError('Original graph changed')
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(SOURCE_VERSION,)).fetchall()
        bindings=db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s',(SOURCE_VERSION,)).fetchall()
        if len(nodes)!=7 or len(bindings)!=17 or any(b['status']!='active' for b in bindings):
            raise ValueError('Source inventory differs')
        for i,(node,(rule,identities)) in enumerate(zip(nodes,EXPECTED_STEPS)):
            sig=json.loads(node['type_signature'])
            actual={b['bound_artifact_fqdn'].rsplit('.',1)[1] for b in bindings if b['node_id']==node['node_id']}
            feeds=[f['sympy'] for f in sig['variable_bindings']['feeds']]
            expected_feeds=["Symbol('pdg0001467')"] if i in [3,4] else []
            if sig['inference_rule_id']!=rule or actual!=identities or feeds!=expected_feeds:
                raise ValueError('Source per-step binding/rule/feed differs')
        for binding in bindings:
            rows=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id JOIN physics_equation_candidates q ON q.candidate_id=e.candidate_id JOIN physics_ingest_snapshots s ON s.snapshot_id=q.snapshot_id WHERE a.fqdn=%s AND v.content_hash=%s',(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
            if len(rows)!=1: raise ValueError('Unique pinned expression snapshot required')
            row=rows[0]; identity=row['source_payload']['id']; raw=row['source_payload']['raw_payload']
            stored_expected = {k:(v.replace(chr(92)*2, chr(92)) if k.startswith('latex_') else v) for k,v in EXPECTED[identity].items()}
            if {k:raw.get(k) for k in stored_expected} != stored_expected:
                raise ValueError('Reviewed stored source fields differ: '+identity)
            evidence=prepare_pdg_evidence(row,symbol_file.read_bytes())
            records[identity]=dict(source_payload_sha256=_digest(row['source_payload']),fresh_source_evidence_sha256=_digest(evidence),equation=EXPECTED[identity])
            pins.append(row['snapshot_payload']['core_file_sha256'])
    if set(records)!=set(EXPECTED) or any(p!=pins[0] for p in pins):
        raise ValueError('Ten source equations with consistent pins required')
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
    result=dict(approved=False,source_nodes=7,source_bindings=17,source_records=records,
        source_file_sha256=pins[0],proof=verify_proof(build_proof()),
        corrections=['Interpret nabla-squared as Cartesian component Laplacian, not a scalar factor.',
                     'Interpret partial derivatives as operators, not partial/t² scalar products.',
                     'Interpret pdg0004621 as imaginary unit and pdg0002718 as exponential.',
                     'Guess is an explicit time-harmonic ansatz, not a consequence of the wave equation.',
                     'Differentiate steps evaluate one then two derivatives already on RHS; do not differentiate the whole equation.',
                     'Explicit time-independent amplitude, real constant omega, positive constant c and mu*epsilon=1/c².',
                     'Electric field units apply to every component; phase is dimensionless and both Helmholtz terms have field/length² units.',
                     'Nonzero exponential cancels even at omega=0; no division by amplitude or frequency.',
                     'Separate Maxwell divergence constraint; scalar Helmholtz is not full electromagnetic validity.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ['scripts/validate_helmholtz_source.py','sciona/physics_ingest/helmholtz_proof.py']})

    historical=json.loads((root/'docs/reviews/helmholtz_execution.json').read_text())['source_proof']
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
