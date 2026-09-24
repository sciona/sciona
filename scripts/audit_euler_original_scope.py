"""Bind the fixed Euler certificate to original provenance and reviewed constants."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
import sympy as sp
from sciona.ghost.symbolic import deserialize_expr
from scripts.validate_euler_source_interpretation import validate
from sciona.physics_ingest.euler_execution import build_euler_execution
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='91eebb1d-83b9-5c66-ab3b-ea44c119a2b9'
ORIGINAL='377af314-2216-5043-808c-ca33523e2beb'
REPLAY='36282a1e-6def-5656-a83c-1971318a6cbe'
EXECUTION='43036a08-5286-5690-a96e-fac7139d10c1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)


def audit(source):
    fresh=validate(ROOT,source/'symbols.cypher',source/'infrules.cypher')
    require(fresh==json.loads((ROOT/'docs/reviews/euler_source_interpretation.json').read_text()),'Interpreted source proof changed')
    require(len(fresh['graphs'])==1 and fresh['graphs'][0]['version_id']==REPLAY,'Exact interpreted source required')
    proof=fresh['graphs'][0];comparisons=[]
    require(proof['passed'] and proof['root_identity_verified'] and len(proof['steps'])==4
        and proof['literal_source_parity'] is False,'Interpreted proof scope differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        old=db.execute('SELECT artifact_id,content_hash FROM artifact_versions WHERE version_id=%s',(ORIGINAL,)).fetchone()
        require(str(old['artifact_id'])==ARTIFACT and old['content_hash']=='634f10ed8d8c7cef29a4454248ef1d0b484ff344daf44ca19da414cef6942212','Original identity differs')
        versions={v:{r['node_id']:json.loads(r['type_signature']) for r in db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s',(v,))} for v in [ORIGINAL,REPLAY]}
        require(set(versions[ORIGINAL])==set(versions[REPLAY]) and len(versions[ORIGINAL])==4,'Complete original four-step scope required')
        mapping=next(iter(versions[REPLAY].values()))['source_expression_substitutions']
        ids=set()
        for key,old in versions[ORIGINAL].items():
            new=versions[REPLAY][key]
            require(new['source_expression_substitutions']==mapping,'Source substitution mapping differs')
            for field in ['inference_rule_id','variable_bindings','source_pdg_step_id']:
                require(old[field]==new[field],'Original rule, feed or step differs')
            for field in ['inputs','outputs']:
                require([mapping.get(x,x) for x in old[field]]==new[field],'Original expression topology differs')
                ids.update(old[field])
        require({mapping.get(x,x) for x in ids}==set(proof['expression_evidence_sha256']),'Complete original expression coverage required')
        for old in sorted(ids):
            new=mapping.get(old,old)
            records={str(r['expression_id']):r for r in db.execute('SELECT expression_id,sympy_srepr FROM artifact_symbolic_expressions WHERE expression_id=ANY(%s::uuid[])',([old,new],))}
            first=deserialize_expr(records[old]['sympy_srepr']);second=deserialize_expr(records[new]['sympy_srepr'])
            require(isinstance(first,sp.Equality) and isinstance(second,sp.Equality),'Source equations required')
            require(all(sp.simplify(a-b)==0 for a,b in zip(first.args,second.args)),'Original parsed/source equation difference')
            comparisons.append(dict(original_expression_id=old,source_expression_id=new,parsed_source_equivalence=True,
                original_expression_sha256=hashlib.sha256(records[old]['sympy_srepr'].encode()).hexdigest(),
                source_evidence_sha256=proof['expression_evidence_sha256'][new]))
        parent=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(EXECUTION,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Approved interpreted derivative required')
        graph=build_euler_execution()
        require(encode_execution_graph(graph)[0]==parent['content_hash'],'Qualified fixed proof graph changed')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=EXECUTION,content_hash=parent['content_hash'],require_execution_envelope=True)==graph,'Stored interpreted graph differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='euler-interpreted-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique interpreted parent approval required')
        approval=rows[0]['details']
        require(approval['publication_tier']==3 and approval['review_source']=='automated','Parent review tier differs')
        for name,digest in approval['evidence_sha256'].items():require(sha(ROOT/'docs/reviews'/name)==digest,'Parent review evidence changed')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,
        source_replay_version_id=REPLAY,approved_execution_version_id=EXECUTION,approved_execution_graph_sha256=parent['content_hash'],
        original_source_equivalence=comparisons,source_steps=4,source_interpretation=proof,
        source_implementation_sha256=fresh['implementation_sha256'],reused_atoms=1,new_atoms=0,literal_source_parity=False,
        scope='Fixed exact Euler identity proof certificate under reviewed source-specific pi and imaginary-unit interpretation; all four source steps retained.',
        reuse='Reusable as an exact certificate component across consumers that require this identity. No parameterized phasor, generic theorem-prover or broad cross-domain numerical capability is claimed.',
        limitations=['Source imaginary-unit identity is marked variable; reviewed constant interpretation and historical failed replay remain explicit.',
            'No runtime inputs. This preserves the original fixed proof scope rather than replacing it with a broader numerical function.',
            'Root identity uses SymPy complex expansion; not a first-principles derivation.',
            'Stored execution, integrity/corruption checks and atomic publication gates remain required before original-identity approval.'],validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_euler_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,expressions=len(report['original_source_equivalence']),steps=4,approved=False)))
