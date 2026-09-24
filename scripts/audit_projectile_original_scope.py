"""Bind immutable projectile source reconstruction to its qualified provider graph."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.validate_projectile_immutable_source import validate
from sciona.physics_ingest.projectile_execution import build_projectile_execution
from sciona.physics_ingest.projectile_proof import SOURCE_VERSION as ORIGINAL,SOURCE_HASH
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='99ea69f3-a28c-51b7-8749-187b5675e7fe'
EXECUTION='8f6d30d7-07bf-52bb-b60a-8e740db53752'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def audit(source):
    proof=validate(ROOT,source/'symbols.cypher',source/'infrules.cypher',source/'expr_and_feed.cypher')
    require(proof['stored_source_equations']==2 and proof['recovered_missing_equations']==2 and len(proof['corrections'])==2,'Complete original source reconstruction required')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(ORIGINAL,)).fetchall()
        require([n['node_id'] for n in nodes]==['pdg_step_1','pdg_step_2'],'Complete original two-step graph required')
        signatures=[json.loads(n['type_signature']) for n in nodes]
        expected=[('111975',['f066dcfc-65d4-58c5-9757-ffdeb3df09b7'],'7927d9c0-45ac-51d6-9750-c3f73a5cd145','2879756'),
            ('111556',['7927d9c0-45ac-51d6-9750-c3f73a5cd145','ac212e18-0e86-5fd8-90cc-a64941222f74'],'bf64ccf1-7142-53ff-a653-29a7fccca938','1975942')]
        for actual,(rule,inputs,output,step) in zip(signatures,expected):
            require(actual['inference_rule_id']==rule and actual['inputs']==inputs and actual['outputs']==[output]
                and actual['output']==output and actual['source_pdg_step_id']==step,'Source rule/expression dependency differs')
            require(actual['variable_bindings']['derivation_id']=='918264' and actual['variable_bindings']['step_id']==step,'Source step identity differs')
        require(signatures[0]['variable_bindings']['feeds']==[dict(feed_id='6050070428',latex='v_{0, x}',sympy="Symbol('pdg0002958')")]
            and signatures[1]['variable_bindings']['feeds']==[],'Source feeds differ')
        require(db.execute('SELECT source_id,target_id FROM artifact_cdg_edges WHERE version_id=%s',(ORIGINAL,)).fetchall()
            ==[dict(source_id='pdg_step_1',target_id='pdg_step_2')],'Source edge differs')
        parent=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(EXECUTION,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Approved projectile implementation required')
        graph=build_projectile_execution();require(encode_execution_graph(graph)[0]==parent['content_hash'],'Qualified graph changed')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=EXECUTION,content_hash=parent['content_hash'],require_execution_envelope=True)==graph,'Stored parent graph differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='projectile-corrected-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent semantic approval required')
        qualification=rows[0]['details'];retained=qualification['corrected_proof']
        require(qualification['publication_tier']==3 and qualification['review_source']=='automated' and qualification['source_parity_claim'] is False,'Parent scope differs')
        # Only the state-independent validator implementation identity differs;
        # every source reconstruction, rule, dimension and proof result must match.
        require({k:v for k,v in proof.items() if k!='implementation_sha256'}=={k:v for k,v in retained.items() if k!='implementation_sha256'},'Immutable source semantics differ from qualified reconstruction')
        for path,digest in retained['implementation_sha256'].items():require(sha(ROOT/path)==digest,'Historical qualified validator changed')
        for name,digest in qualification['evidence_sha256'].items():require(sha(ROOT/'docs/reviews'/name)==digest,'Qualified parent evidence changed')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,original_content_hash=SOURCE_HASH,
        approved_execution_version_id=EXECUTION,approved_execution_graph_sha256=parent['content_hash'],source_steps=2,source_bindings=5,
        recovered_missing_equations=2,explicit_source_corrections=2,source_proof=proof,source_parity_claim=False,reused_atoms=1,new_atoms=0,
        scope='Corrected ideal-projectile elapsed time and height at a horizontal coordinate, preserving both source steps and all four source equations.',
        limitations=['Two missing equations are recovered from the exact pinned public source file. Initial vertical velocity and linear gravity replace explicit source symbolic defects.',
            'Caller establishes one Cartesian frame, SI units, constant nonnegative downward gravity and ideal no-drag motion. No collision, terrain, wind or varying-gravity model.',
            'Nonzero horizontal velocity and nonnegative elapsed time required; either horizontal direction and the zero-gravity inertial limit are supported.',
            'Original source and historical draft-only validator remain preserved; stored original-identity execution and publication gates remain required.'],validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_projectile_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,source_steps=2,recovered_equations=2,source_corrections=2,approved=False)))
