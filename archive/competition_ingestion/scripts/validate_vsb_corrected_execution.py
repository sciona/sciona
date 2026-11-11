"""Execute the stored corrected candidate through the production converter."""
import json
from unittest.mock import patch
from scripts import stage_vsb_corrected_intake as intake
from scripts import validate_decomposed_vsb_graph as execution
from scripts.review_decomposed_vsb_semantics import packages
from sciona.services.execution_graph_codec import encode_execution_graph


def validate():
    plan,providers=intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph=intake.check_candidate(db,plan,providers)
    if encode_execution_graph(graph)[0]!=plan['graph_sha256']:
        raise ValueError('Stored graph digest differs')
    paths=['scripts/validate_vsb_corrected_execution.py',
        'scripts/stage_vsb_corrected_intake.py',
        'scripts/decomposed_vsb_approved_provider_state.py',
        'scripts/validate_decomposed_vsb_graph.py',
        'scripts/review_decomposed_vsb_semantics.py',
        'docs/reviews/vsb_corrected_intake_plan.json',
        'docs/reviews/vsb_approved_provider_state.json']
    before={p:intake.sha(intake.ROOT/p) for p in paths}
    installed=packages()
    qualified=json.loads((intake.ROOT/'docs/reviews/decomposed_vsb_semantics.json').read_text())
    if installed!=qualified['installed_packages']:raise ValueError('Qualified runtime changed')
    # Use the unchanged full reference-parity harness, supplying only the graph
    # actually materialized from catalog rows instead of the local builder.
    with patch.object(execution,'build_decomposed_vsb_graph',return_value=graph):
        result=execution.validate()
    baseline=qualified['execution']
    for key in ('cases','provider_sha256','implementation_sha256','nodes','edges','distinct_providers'):
        if result[key]!=baseline[key]:raise ValueError('Complete qualified execution differs: '+key)
    if result['graph_sha256']!=plan['graph_sha256']:
        raise ValueError('Runner executed a different graph')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if intake.check_candidate(db,plan,providers)!=graph:raise ValueError('Stored graph changed during execution')
    if before!={p:intake.sha(intake.ROOT/p) for p in paths} or installed!=packages():
        raise ValueError('Execution evidence changed')
    return dict(format='vsb-corrected-stored-execution.v1',passed=True,approved=False,
        catalog_mutations=0,synthetic_only=True,stored_catalog_graph_executed=True,
        artifact_id=plan['artifact_id'],version_id=plan['version_id'],graph_sha256=plan['graph_sha256'],
        execution=result,evidence_sha256=before,installed_packages=installed,
        limits=['Complete stored 26-node graph through the production converter and native runner; original approval remains pending.',
                'Two synthetic cases execute all125 models with full source stopping limits, exact numeric callables and final signal decisions; reference equality does not certify historical accuracy or cross-domain generalization.'])


if __name__=='__main__':
    result=validate()
    (intake.ROOT/'docs/reviews/vsb_corrected_intake_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,stored_graph=True,nodes=result['execution']['nodes'],cases=len(result['execution']['cases']),approved=False)))
