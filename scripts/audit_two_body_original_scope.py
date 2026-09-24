"""Verify original two-body source scope through its explicit one-pin repair."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.validate_repaired_source_graph import validate as select
from scripts.validate_two_body_corrected_proof import validate_source
from sciona.physics_ingest.two_body_execution import build_two_body_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='d20344b6-9c2b-52ae-accb-705cb6717e24'
ORIGINAL='ffb043f3-05a3-5ca6-9d94-87a17ff512d3'
REPLAY='76274cb1-ebf6-533d-a777-b505190f3d17'
EXECUTION='987a2855-dca8-52e5-a6d5-47b6f7736b4e'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def audit(source):
    selection=select(ROOT,source/'symbols.cypher',source/'infrules.cypher')
    proof=validate_source(selection,(source/'symbols.cypher').read_bytes())
    require(selection['source_graph_version_id']==REPLAY and proof['selected_source_equations_matched']==13
        and proof['selected_source_premises_matched']==6 and proof['source_parity_claim'] is False,'Complete corrected source scope required')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        original=db.execute('SELECT artifact_id::text,content_hash FROM artifact_versions WHERE version_id=%s',(ORIGINAL,)).fetchone()
        require(original==dict(artifact_id=ARTIFACT,content_hash='4d2810f058da2f5d3ec97b8de34b7219dee8b7c1715d99d37f3544fe3e9df9c4'),'Original identity differs')
        repair=db.execute('SELECT content_hash,derives_from::text FROM artifact_versions WHERE version_id=%s',(REPLAY,)).fetchone()
        require(repair==dict(content_hash=selection['source_graph_content_hash'],derives_from=ORIGINAL),'Repair lineage differs')
        records={}
        for version in [ORIGINAL,REPLAY]:
            nodes=[dict(node_id=r['node_id'],signature=json.loads(r['type_signature'])) for r in db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(version,))]
            edges=db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id',(version,)).fetchall()
            bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id,bound_artifact_fqdn',(version,)).fetchall()
            records[version]=(nodes,edges,bindings)
        old,new=records[ORIGINAL],records[REPLAY]
        require(old[:2]==new[:2] and len(old[0])==13 and len(old[2])==len(new[2])==31,'Original full source topology differs')
        changes=[dict(original=a,repaired=b) for a,b in zip(old[2],new[2]) if a!=b]
        require(len(changes)==1,'Exactly one repaired binding required')
        change=changes[0]
        for r in [change['original'],change['repaired']]:
            require(r['node_id']=='pdg_step_2' and r['bound_artifact_fqdn']=='physics.pdg.remote_wave.equation.1292735067' and r['status']=='active','Repair identity differs')
        require(change['original']['bound_version_content_hash']=='8d7dec2c3f13e7dcc89abc31b3d2e1539fff56acd55b0d78dfd7b36f7feca4cf'
            and change['repaired']['bound_version_content_hash']=='baa1e77fdcad1911fe30ad4107edf241fbe0fa4f88e149862b99a5cf06eadf48','Explicit repair hashes differ')
        parent=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(EXECUTION,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Qualified two-body implementation required')
        graph=build_two_body_execution();require(encode_execution_graph(graph)[0]==parent['content_hash'],'Qualified execution changed')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=EXECUTION,content_hash=parent['content_hash'],require_execution_envelope=True)==graph,'Stored parent graph differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='two_body-corrected-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent semantic approval required')
        for name,digest in rows[0]['details']['evidence_sha256'].items():require(sha(ROOT/'docs/reviews'/name)==digest,'Retained parent evidence changed')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,source_replay_version_id=REPLAY,
        approved_execution_version_id=EXECUTION,approved_execution_graph_sha256=parent['content_hash'],source_steps=13,source_premises=6,
        repaired_binding=change,original_topology_preserved=True,source_selection=selection,corrected_proof=proof,source_parity_claim=False,reused_atoms=1,new_atoms=0,
        scope='Corrected complete circular Newtonian two-body period from positive SI separation, both masses and explicit gravitational constant.',
        limitations=['Original stale binding and source-rule defects remain historical evidence. Corrected substitutions, explicit mass division and separation premise are required.',
            'Caller establishes isolated Newtonian point masses, circular orbit and consistent SI units; no orbit classifier or relativistic, extended-body or perturbed-orbit extension.',
            'Exact source pi identity is separately interpreted; numerical provider uses binary64 pi and high-precision intermediates.',
            'Original-identity stored execution, integrity/corruption and publication transaction checks remain required.'],validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_two_body_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,source_steps=13,premises=6,repaired_bindings=1,approved=False)))
