"""Execute the stored corrected candidate through the production converter."""
import json
from unittest.mock import patch
from scripts import stage_plasticc_corrected_intake as intake
from scripts import validate_decomposed_plasticc_graph_v2 as execution
from scripts.review_decomposed_plasticc_semantics import packages
from sciona.services.execution_graph_codec import encode_execution_graph


def validate():
    plan,providers=intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph=intake.check_candidate(db,plan,providers)
    if encode_execution_graph(graph)[0]!=plan['graph_sha256']:
        raise ValueError('Stored graph digest differs')
    paths=['scripts/validate_plasticc_corrected_execution.py',
        'scripts/stage_plasticc_corrected_intake.py',
        'scripts/decomposed_plasticc_approved_provider_state.py',
        'scripts/validate_decomposed_plasticc_graph_v2.py',
        'scripts/review_decomposed_plasticc_semantics.py',
        'docs/reviews/plasticc_corrected_intake_plan.json',
        'docs/reviews/plasticc_approved_provider_state.json']
    before={p:intake.sha(intake.ROOT/p) for p in paths}
    installed=packages()
    from scripts.review_decomposed_plasticc_interfaces import pinned
    qualified=pinned('docs/reviews/decomposed_plasticc_semantics.json', '055a28a9')
    if installed!=qualified['installed_packages']:raise ValueError('Qualified runtime changed')
    # Use the unchanged full reference-parity harness, supplying only the graph
    # actually materialized from catalog rows instead of the local builder.
    with patch.object(execution,'build_decomposed_plasticc_graph',return_value=graph):
        result=execution.validate()
    baseline=pinned('docs/reviews/decomposed_plasticc_graph_execution_v2.json', 'b5a0a8be')
    if baseline['graph_sha256']!=qualified['graph_digest'] or baseline['cases']!=qualified['graph_cases']:
        raise ValueError('Semantic and graph qualification differ')
    for key in ('cases','provider_sha256','implementation_sha256','nodes','edges','distinct_providers'):
        if result[key]!=baseline[key]:raise ValueError('Complete qualified execution differs: '+key)
    if result['graph_sha256']!=plan['graph_sha256']:
        raise ValueError('Runner executed a different graph')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if intake.check_candidate(db,plan,providers)!=graph:raise ValueError('Stored graph changed during execution')
    if before!={p:intake.sha(intake.ROOT/p) for p in paths} or installed!=packages():
        raise ValueError('Execution evidence changed')
    return dict(format='plasticc-corrected-stored-execution.v1',passed=True,approved=False,
        catalog_mutations=0,synthetic_only=True,stored_catalog_graph_executed=True,
        artifact_id=plan['artifact_id'],version_id=plan['version_id'],graph_sha256=plan['graph_sha256'],
        execution=result,evidence_sha256=before,installed_packages=installed,
        limits=['Complete stored 26-node graph through the production converter and native runner; original approval remains pending.',
                'Two synthetic weighting cases execute all three configured folds, one augmentation attempt per original and a 60-tree ceiling, with exact numeric callables and final astronomy class policy; reference equality does not certify historical scale, accuracy or cross-domain generalization.'])


if __name__=='__main__':
    result=validate()
    (intake.ROOT/'docs/reviews/plasticc_corrected_intake_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,stored_graph=True,nodes=result['execution']['nodes'],cases=len(result['execution']['cases']),approved=False)))
