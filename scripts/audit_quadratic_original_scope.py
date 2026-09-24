"""Bind a corrected quadratic scope to immutable defective original provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
import sympy as sp
from scripts.audit_physics_branch_steps import validate as diagnose
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.quadratic_corrected_proof import build_proof,verify_proof
from sciona.physics_ingest.quadratic_execution import build_quadratic_execution
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='41b208a7-9b47-5b0e-920e-c2480a73c193'
ORIGINAL='b8971460-66c8-50be-a7ce-5532d3133860'
REPLAY='40ae98f0-f033-5861-870c-6d6408edbd2b'
EXECUTION='25734209-2f61-59da-9be9-70020245ae1d'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def audit(source):
    diagnostic=diagnose(ROOT,source/'symbols.cypher',source/'infrules.cypher')
    require(diagnostic==json.loads((ROOT/'docs/reviews/physics_branch_step_diagnostics.json').read_text()),'Source diagnostics changed')
    rows=[r for r in diagnostic['graphs'] if r['version_id']==REPLAY]
    require(len(rows)==1,'Exact quadratic source diagnostic required')
    defects=[s for s in rows[0]['steps'] if 'counterexample' in s]
    require({s['node_id'] for s in defects}=={'pdg_step_1','pdg_step_2','pdg_step_8','pdg_step_9'},'All four original defects must remain explicit')
    proof=verify_proof(build_proof());comparisons=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        original=db.execute('SELECT artifact_id,content_hash FROM artifact_versions WHERE version_id=%s',(ORIGINAL,)).fetchone()
        require(str(original['artifact_id'])==ARTIFACT and original['content_hash']=='aa80be9017c41ca0faf04c039513ac894834159b74afc36a1285a1f5a2396aa9','Original identity differs')
        versions={}
        for version in [ORIGINAL,REPLAY]:
            versions[version]={r['node_id']:json.loads(r['type_signature']) for r in db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(version,))}
        require(set(versions[ORIGINAL])==set(versions[REPLAY]) and len(versions[ORIGINAL])==11,'Complete source topology required')
        mapping=next(iter(versions[REPLAY].values()))['source_expression_substitutions']
        require(set(mapping.values())==set(rows[0]['expression_evidence_sha256']),'Complete source expressions required')
        for key,old in versions[ORIGINAL].items():
            new=versions[REPLAY][key]
            require(new['source_expression_substitutions']==mapping,'Expression mapping differs')
            for field in ['inference_rule_id','variable_bindings','source_pdg_step_id']:
                require(old[field]==new[field],'Original rule, feed or source step changed')
            for field in ['inputs','outputs']:
                require([mapping[x] for x in old[field]]==new[field],'Original expression topology changed')
        for old,new in sorted(mapping.items()):
            records={str(r['expression_id']):r for r in db.execute('SELECT expression_id,sympy_srepr FROM artifact_symbolic_expressions WHERE expression_id=ANY(%s::uuid[])',([old,new],))}
            first=deserialize_expr(records[old]['sympy_srepr']);second=deserialize_expr(records[new]['sympy_srepr'])
            require(isinstance(first,sp.Equality) and isinstance(second,sp.Equality),'Original equations required')
            equivalent=all(sp.simplify(a-b)==0 for a,b in zip(first.args,second.args))
            correction=None
            if not equivalent:
                a,b,c,x=sp.symbols('a b c x')
                if old=='0ff7da4c-8c16-5fab-9419-4c97866c03d0':
                    expected=build_proof().divided
                    plain=lambda e:e.xreplace({v:sp.Symbol(str(v)) for v in e.free_symbols})
                    require(all(sp.simplify(a-b)==0 for a,b in zip(first.args,plain(expected).args)),
                        'Original divided equation does not retain corrected b/a')
                    require(sp.simplify(second.lhs-(x*x+x+c/a))==0 and second.rhs==0,
                        'Unexpected normalized source division defect')
                    correction='Original parsed division retains b/a; normalized upstream source loses b/a. Corrected proof retains b/a.'
                else:
                    malformed={
                        'cc3aa795-747d-57f2-9947-8d6ee06e683b':x*x+sp.Function('x')(b/a)+(b/(2*a))**2,
                        'ddbfe37a-42fa-5e16-b127-c1282750bd85':x*x+2*sp.Function('x')(b/(2*a))+(b/(2*a))**2,
                    }
                    require(old in malformed,'Unreviewed original/source equation difference')
                    require(sp.simplify(first.lhs-malformed[old])==0
                        and sp.simplify(first.rhs-(x+b/(2*a))**2)==0,'Unexpected original implicit-product parse')
                    require(sp.simplify(second.lhs-(x*x+b*x/a+(b/(2*a))**2))==0
                        and sp.simplify(second.rhs-(x+b/(2*a))**2)==0,'Unexpected source square identity')
                    correction='Original parser interprets implicit multiplication as an undefined x function; normalized source and corrected proof use multiplication. No original parsed-equation equivalence claim.'
            comparisons.append(dict(original_expression_id=old,source_expression_id=new,parsed_source_equivalence=equivalent,explicit_discrepancy=correction,
                original_expression_sha256=hashlib.sha256(records[old]['sympy_srepr'].encode()).hexdigest(),
                source_evidence_sha256=rows[0]['expression_evidence_sha256'][new]))
        require(sum(not c['parsed_source_equivalence'] for c in comparisons)==3,'Expected three explicit original/source discrepancies')
        parent=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(EXECUTION,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Approved corrected derivative required')
        graph=build_quadratic_execution()
        require(encode_execution_graph(graph)[0]==parent['content_hash'],'Corrected numerical graph changed')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=EXECUTION,content_hash=parent['content_hash'],require_execution_envelope=True)==graph,'Stored corrected derivative differs')
        approvals=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='quadratic-corrected-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(approvals)==1,'Unique corrected approval required')
        approval=approvals[0]['details']
        require(approval['corrected_proof']==proof and approval['source_parity_claim'] is False and approval['publication_tier']==3,'Corrected proof review differs')
        for name,digest in approval['evidence_sha256'].items():require(sha(ROOT/'docs/reviews'/name)==digest,'Qualified parent evidence changed')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,
        source_replay_version_id=REPLAY,approved_execution_version_id=EXECUTION,approved_execution_graph_sha256=parent['content_hash'],
        original_source_equivalence=comparisons,source_steps=11,verified_source_counterexamples=defects,explicit_original_source_discrepancies=3,
        corrected_proof=proof,source_parity_claim=False,reused_atoms=1,new_atoms=0,
        scope='Explicitly corrected complete real quadratic solution with ordered lower and upper roots; defective original proof remains immutable provenance.',
        reuse='Caller supplies dimensionally consistent coefficients; generic algebra applies across domains without embedding a dataset, physical interpretation or fitted model.',
        limitations=['Real finite coefficients, identical nonempty shapes, nonzero leading coefficient, nonnegative discriminant and finite representable roots.',
            'Preserve both alternative roots, including repeated and zero roots; no linear fallback or complex-root extension.',
            'No claim that the original defective derivation is valid or that numerical correctness establishes physical applicability.',
            'Fresh stored execution, cross-domain synthetic examples, corruption and transaction gates are still required before original-identity activation.'],
        source_diagnostics=rows[0],source_implementation_sha256=diagnostic['implementation_sha256'],validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_quadratic_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,source_expressions=len(report['original_source_equivalence']),source_steps=11,counterexamples=4,approved=False)))
