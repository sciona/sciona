"""Verify production selection equals the fully executed stored graph, without rerunning it."""
import argparse
import json
from sciona.services.catalog_artifact_retrieval import CatalogMacroArtifactRetriever
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from scripts import promote_openvaccine_corrected_intakes as publisher
from scripts import stage_openvaccine_corrected_intakes as staging


def verify(target):
    intake=staging.Intake(target)
    plan,providers,qualification=publisher.review(target)
    with staging.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        expected=intake.check_candidate(db,plan,providers,approved=True)
        rows=db.execute('SELECT artifact_id,artifact_kind,fqdn,technical_description,domain_tags FROM catalog_artifacts_served WHERE artifact_id=%s',(plan['artifact_id'],)).fetchall()
        versions=db.execute('SELECT version_id,semver,content_hash,trust_tier FROM artifact_versions WHERE artifact_id=%s AND is_latest',(plan['artifact_id'],)).fetchall()
        if len(rows)!=1 or len(versions)!=1 or str(versions[0]['version_id'])!=plan['version_id']:
            raise ValueError('Production selection differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(plan['fqdn'],)).fetchone()['d']
        candidate=CatalogMacroArtifactRetriever(None)._candidate_from_document(row=rows[0],document=document,version=versions[0])
        if candidate.cdg!=expected or candidate.content_hash!=plan['graph_sha256'] or candidate.trust_tier!=3:
            raise ValueError('Production graph conversion differs')
        if candidate.description!=publisher.DESCRIPTION or candidate.conceptual_summary!=publisher.PLAIN or candidate.verified_leaf_coverage!=1:
            raise ValueError('Production metadata differs')
        parameters=db.execute('SELECT name,type_desc,required,technical_description,constraints_json FROM artifact_parameters WHERE artifact_id=%s AND version_id=%s',
            (plan['artifact_id'],plan['version_id'])).fetchall()
        if parameters!=[dict(name='payload',type_desc='dict',required=True,technical_description=publisher.PARAMETER_CONTRACT,
            constraints_json=dict(parameter_contract=publisher.PARAMETER_CONTRACT,output_contract=publisher.OUTPUT_CONTRACT,
                validity_bounds=qualification['limitations'],review_status='automated_tier_3_pass_with_limits'))]:
            raise ValueError('Complete payload or output contract missing')
        if not db.execute('SELECT artifact_is_publishable(%s) AS yes',(plan['artifact_id'],)).fetchone()['yes']:
            raise ValueError('Native CDG eligibility missing')
        evidence=db.execute('SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s AND passed AND status=%s',(plan['version_id'],publisher.runner_name(target),'completed')).fetchall()
        if len(evidence)!=1 or evidence[0]['details']!=dict(qualification=qualification,publisher_sha256=staging.sha(publisher.__file__),description=publisher.DESCRIPTION,plain_description=publisher.PLAIN):
            raise ValueError('Publication evidence differs')
        digest,nodes,edges=encode_execution_graph(candidate.cdg)
        if digest!=plan['graph_sha256'] or decode_execution_graph(nodes,edges,digest)!=expected:
            raise ValueError('Selected execution envelope differs')
    return dict(format='openvaccine-corrected-served-selection.v1',target=target,passed=True,approved=True,trust_tier=3,
        production_candidate_conversion=True,selected_graph_equals_fully_executed_candidate=True,
        full_execution_report_sha256=qualification['execution_sha256'],catalog_mutations=0,
        original_artifact_id=plan['artifact_id'],selected_version_id=plan['version_id'],graph_sha256=digest,
        source_histories_preserved=True,approved_providers=6,verifier_sha256=staging.sha(__file__),
        publisher_sha256=staging.sha(publisher.__file__),
        limits='Selection checks the served SQL view, latest predicate, document RPC, production converter and exact qualified execution envelope. No post-publication model rerun, HTTP transport or semantic-search ranking claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',choices=list(staging.TARGETS))
    target=parser.parse_args().target
    result=verify(target)
    staging.Intake(target).report_path('served_selection').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
