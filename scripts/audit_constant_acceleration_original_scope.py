"""Bind immutable constant_acceleration source reconstruction to its qualified provider graph."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.validate_constant_acceleration_immutable_source import validate
from sciona.physics_ingest.constant_acceleration_execution import build_constant_acceleration_execution
from sciona.physics_ingest.constant_acceleration_proof import SOURCE_VERSION as ORIGINAL,SOURCE_HASH
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='95f290e0-a920-5569-bc87-ab25e62f6562'
EXECUTION='44918f03-3961-5bf6-8335-b3256abdb339'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def audit(source):
    proof=validate(ROOT,source/'symbols.cypher',source/'infrules.cypher',source/'expr_and_feed.cypher')
    require(proof['source_nodes']==23 and proof['source_bindings']==51 and len(proof['source_records'])==23
        and proof['recovered_missing_equations']==['3411994811'] and len(proof['corrections'])==6,'Complete source reconstruction required')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        parent=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(EXECUTION,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Approved constant_acceleration implementation required')
        graph=build_constant_acceleration_execution();require(encode_execution_graph(graph)[0]==parent['content_hash'],'Qualified graph changed')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=EXECUTION,content_hash=parent['content_hash'],require_execution_envelope=True)==graph,'Stored parent graph differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='constant_acceleration-corrected-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent semantic approval required')
        qualification=rows[0]['details'];retained=qualification['corrected_proof']
        require(qualification['publication_tier']==3 and qualification['review_source']=='automated' and qualification['source_parity_claim'] is False,'Parent scope differs')
        # Only the state-independent validator implementation identity differs;
        # every source reconstruction, rule, dimension and proof result must match.
        require({k:v for k,v in proof.items() if k!='implementation_sha256'}=={k:v for k,v in retained.items() if k!='implementation_sha256'},'Immutable source semantics differ from qualified reconstruction')
        for path,digest in retained['implementation_sha256'].items():require(sha(ROOT/path)==digest,'Historical qualified validator changed')
        for name,digest in qualification['evidence_sha256'].items():require(sha(ROOT/'docs/reviews'/name)==digest,'Qualified parent evidence changed')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,original_content_hash=SOURCE_HASH,
        approved_execution_version_id=EXECUTION,approved_execution_graph_sha256=parent['content_hash'],source_steps=23,source_bindings=51,
        recovered_missing_equations=1,explicit_source_corrections=6,source_proof=proof,source_parity_claim=False,reused_atoms=1,new_atoms=0,
        scope='Corrected signed constant-acceleration motion covering all 23 source steps and 24 source expressions.',
        limitations=['One missing equation and the damaged squared-velocity AST are reconstructed from exact pinned public source bytes.',
            'Finite signed one-dimensional motion, SI units, constant acceleration and nonnegative elapsed time. Displacement is not path length.',
            'Zero acceleration and zero time use independently proved polynomial extensions; zero-time average velocity is defined by continuity.',
            'Squared velocity does not determine a unique signed inverse. Variable acceleration is outside the contract.',
            'Original source and historical draft-only validator remain preserved; stored original-identity execution and publication gates remain required.'],validator_sha256=sha(__file__))



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_constant_acceleration_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,source_steps=23,recovered_equations=1,source_corrections=6,approved=False)))
