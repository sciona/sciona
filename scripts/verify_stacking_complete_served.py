"""Execute the freshly served complete graph with the qualified native validator.

Only graph acquisition and success-report destination are redirected. Every
numerical learner and all validation assertions run unchanged. Original native
qualification evidence is preserved byte for byte.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts import validate_stacking_complete_graph as native
from scripts.check_stacking_complete_catalog import ROOT, ORIGINAL_FQDN, qualification, require, sha
from scripts.promote_stacking_complete import check_publication, DESCRIPTION, PLAIN_DESCRIPTION
from scripts.review_stacking_complete_publication import review


def selection():
    scope,specs=qualification(); qualified=review()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_publication(db,scope,specs,qualified)
        rows=db.execute('SELECT artifact_id,artifact_kind,fqdn,technical_description,domain_tags '
                        'FROM catalog_artifacts_served WHERE artifact_id=%s',(scope['artifact_id'],)).fetchall()
        require(len(rows)==1,'One original served CDG required')
        versions=db.execute('SELECT version_id,semver,content_hash,trust_tier FROM artifact_versions '
                            'WHERE artifact_id=%s AND is_latest',(scope['artifact_id'],)).fetchall()
        require(len(versions)==1 and str(versions[0]['version_id'])==scope['corrected_version_id'],
                'Served latest version differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(ORIGINAL_FQDN,)).fetchone()['d']
        candidate=CatalogMacroArtifactRetriever(None)._candidate_from_document(row=rows[0],document=document,version=versions[0])
        require(candidate.content_hash==scope['graph_sha256'] and candidate.trust_tier==3
                and encode_execution_graph(candidate.cdg)[0]==scope['graph_sha256'],'Served runtime graph differs')
        require(candidate.description==DESCRIPTION and candidate.conceptual_summary==PLAIN_DESCRIPTION
                and candidate.verified_leaf_coverage==1,'Served descriptions or coverage differ')
    return candidate.cdg,dict(passed=True,artifact_id=scope['artifact_id'],version_id=scope['corrected_version_id'],
        graph_sha256=scope['graph_sha256'],tier=3,providers=27,nodes=62,edges=198,
        production_candidate_converter=True,corrected_descriptions_verified=True)


def verify():
    graph,before=selection()
    evidence=ROOT/'docs/reviews/competition_otto_decomposed_complete_execution.json'
    retained=evidence.read_bytes(); captured=[]; original_write=Path.write_text
    code_before=sha(__file__)

    def redirect_success(path,text,*args,**kwargs):
        if path.resolve()==evidence.resolve():
            captured.append(json.loads(text))
            return len(text)
        return original_write(path,text,*args,**kwargs)

    # Graph source is the actual production candidate above. No learner, model
    # count, fixture, numerical function, oracle or runner is substituted here.
    with patch.object(native,'build_complete_graph',return_value=graph), patch.object(Path,'write_text',redirect_success):
        native.main()
    require(len(captured)==1 and captured[0]['status']=='passed','Native served validation did not finish')
    require(evidence.read_bytes()==retained,'Original native evidence changed')
    after_graph,after=selection()
    require(after==before and after_graph==graph,'Served selection changed during execution')
    require(sha(__file__)==code_before,'Served verifier changed during execution')
    return dict(passed=True,read_only_catalog=True,catalog_mutations=0,tier=3,selection=before,
        execution=captured[0],verifier_sha256=code_before,native_validator_sha256=sha(native.__file__),
        retrieval_sha256=sha(ROOT/'sciona/services/catalog_artifact_retrieval.py'),
        original_native_evidence_preserved=True,
        limitations=['Full native synthetic execution through fresh served SQL selection and production candidate conversion.',
                     'HTTP transport, semantic search ranking, empirical effectiveness and clean installation are unqualified.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selection-only',action='store_true');args=parser.parse_args()
    if args.selection_only:
        _,result=selection(); name='competition_otto_decomposed_served_selection.json'
    else:
        result=verify(); name='competition_otto_decomposed_served_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,selection_only=args.selection_only,tier=3)))
