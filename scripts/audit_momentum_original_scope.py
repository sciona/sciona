"""Bind immutable momentum source reconstruction to its qualified provider graph."""
import argparse
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.validate_momentum_immutable_source import validate
from sciona.physics_ingest.momentum_execution import build_momentum_execution
from sciona.physics_ingest.momentum_vector_proof import SOURCE_VERSION as ORIGINAL,SOURCE_HASH
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='279f6f24-1db8-5ee3-a09f-f0283ca33ebb'
EXECUTION='b5f856e1-441e-5acd-8562-5546b7dd0b6c'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def audit(source):
    proof=validate(ROOT,source/'symbols.cypher',source/'infrules.cypher')
    require(proof['selected_source_bindings']==12 and proof['vector_equations_matched']==7 and proof['explicitly_reconstructed_dot_products']==1,'Complete original source reconstruction required')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        parent=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(EXECUTION,)).fetchone()
        require(parent and parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Approved momentum implementation required')
        graph=build_momentum_execution();require(encode_execution_graph(graph)[0]==parent['content_hash'],'Qualified graph changed')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document,version_id=EXECUTION,content_hash=parent['content_hash'],require_execution_envelope=True)==graph,'Stored parent graph differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='momentum-corrected-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent semantic approval required')
        qualification=rows[0]['details'];retained=qualification['corrected_proof']
        require(qualification['publication_tier']==3 and qualification['review_source']=='automated' and qualification['source_parity_claim'] is False,'Parent scope differs')
        # Only the state-independent validator implementation identity differs;
        # every source reconstruction, rule, dimension and proof result must match.
        require({k:v for k,v in proof.items() if k!='implementation_sha256'}=={k:v for k,v in retained.items() if k!='implementation_sha256'},'Immutable source semantics differ from qualified reconstruction')
        for path,digest in retained['implementation_sha256'].items():require(sha(ROOT/path)==digest,'Historical qualified validator changed')
        for name,digest in qualification['evidence_sha256'].items():require(sha(ROOT/'docs/reviews'/name)==digest,'Qualified parent evidence changed')
    return dict(passed=True,approved=False,catalog_mutations=0,artifact_id=ARTIFACT,original_version_id=ORIGINAL,original_content_hash=SOURCE_HASH,
        approved_execution_version_id=EXECUTION,approved_execution_graph_sha256=parent['content_hash'],source_steps=5,source_bindings=12,
        source_equations_matched=7,explicitly_reconstructed_dot_products=1,source_proof=proof,source_parity_claim=False,reused_atoms=1,new_atoms=0,
        scope='Real 3D recoil vector and squared Euclidean norm under explicit momentum-conservation premises; incomplete final source dot-product expression reconstructed from exact reviewed LaTeX.',
        limitations=['Both vectors use one orthonormal frame and consistent SI momentum units; conservation and physical applicability are caller premises.',
            'No full Compton event, energy, dispersion or scattering-angle solver. Exact converted-input differences are squared before independent output rounding.',
            'Original source and historical draft-only validator are preserved. The new validator checks immutable version identity independently of current artifact status.',
            'Stored original-identity execution, corruption and publication gates remain required.'],validator_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/physics_momentum_original_scope.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,source_steps=5,matched_equations=7,reconstructed_dot_products=1,approved=False)))
