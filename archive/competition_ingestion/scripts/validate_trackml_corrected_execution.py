"""Execute the stored original candidate through the full source-parity harness."""
import json
from unittest.mock import patch

from scripts import stage_trackml_corrected_intake as intake
from scripts import validate_decomposed_trackml_cell_graph as execution
from scripts.review_trackml_cell_adapter_semantics import packages
from sciona import decomposed_trackml_cell_graph as builder
from sciona.services.execution_graph_codec import encode_execution_graph


def validate():
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph = intake.check_candidate(db, plan, providers)
    if encode_execution_graph(graph)[0] != plan['graph_sha256']:
        raise ValueError('Stored candidate graph differs')
    paths = ['scripts/validate_trackml_corrected_execution.py',
        'scripts/stage_trackml_corrected_intake.py',
        'scripts/decomposed_trackml_approved_provider_state.py',
        'docs/reviews/trackml_corrected_intake_plan.json',
        'docs/reviews/trackml_approved_provider_state.json']
    before = {p:intake.sha(intake.ROOT/p) for p in paths}
    installed = packages()
    baseline = json.loads((intake.ROOT/'docs/reviews/decomposed_trackml_cell_graph_execution.json').read_text())
    # Replace only the graph constructor imported by the existing execution harness.
    # Its four uncached native runs and exact callable checks remain unchanged.
    with patch.object(builder, 'build_decomposed_trackml_cell_graph', return_value=graph):
        result = execution.validate()
    for key in baseline:
        if key != 'graph_sha256' and result[key] != baseline[key]:
            raise ValueError('Qualified execution differs: '+key)
    if result['graph_sha256'] != plan['graph_sha256']:
        raise ValueError('Runner executed a different graph')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if intake.check_candidate(db, plan, providers) != graph:
            raise ValueError('Stored graph changed during execution')
    if installed != packages() or before != {p:intake.sha(intake.ROOT/p) for p in paths}:
        raise ValueError('Execution dependencies changed')
    return dict(format='trackml-corrected-stored-execution.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, stored_catalog_graph_executed=True,
        artifact_id=plan['artifact_id'], version_id=plan['version_id'], graph_sha256=plan['graph_sha256'],
        execution=result, evidence_sha256=before, installed_packages=installed,
        limits=['Complete stored 28-node candidate through production conversion and native execution; original publication remains pending.',
                'Synthetic source parity includes configured rounds, supplied maps and optional cells; no historical accuracy or calibration-learning claim.'])


if __name__ == '__main__':
    result = validate()
    (intake.ROOT/'docs/reviews/trackml_corrected_intake_execution.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, stored_graph=True, nodes=28, approved=False)))
