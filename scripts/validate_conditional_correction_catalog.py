"""Execute the exact draft retrieved from the catalog with synthetic inputs."""
import asyncio
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.visualizer import runner
from scripts.plan_conditional_correction_promotion import plan
from scripts.review_conditional_correction import ROOT, require, sha


def validate():
    proposed = plan()
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        document = db.execute('SELECT get_artifact_document(%s) AS d', (proposed['fqdn'],)).fetchone()['d']
        bindings = db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status FROM '
            'artifact_cdg_bindings WHERE version_id=%s', (proposed['version_id'],)).fetchall()
        expected = {b['node_id']: (b['fqdn'], b['content_hash'], 'active') for b in proposed['bindings']}
        require({b['node_id']: (b['bound_artifact_fqdn'], b['bound_version_content_hash'], b['status'])
                 for b in bindings} == expected, 'Stored bindings differ')
    graph = _artifact_document_to_cdg(document, version_id=proposed['version_id'],
                                     content_hash=proposed['graph_sha256'], require_execution_envelope=True)
    require(encode_execution_graph(graph)[0] == proposed['graph_sha256'], 'Retrieved graph differs')
    runner._ensure_atoms_imported()
    cases = []
    for domain, base, scale in [('manufacturing_dimensions', 2., .01), ('energy_forecasts', 500., 20.)]:
        calibration = base + scale * np.arange(1., 6.)
        predictions = base + scale * np.arange(8., 11.)
        payload = dict(observed=calibration + scale*np.array([2., 4., -1., -3., 999.]),
            calibration_predictions=calibration, calibration_probabilities=np.array([.8, .9, .1, .2, .5]),
            predictions=predictions, probabilities=np.array([.7, .3, .5]), threshold=.5)
        for invalid in [False, True]:
            selected = dict(payload)
            if invalid:
                selected['probabilities'] = np.array([1.1, .3, .5])
            captured = {}
            def capture(directory, node, name, value):
                if node == 'correct' and name == 'out_corrected_predictions':
                    captured['output'] = value
            with tempfile.TemporaryDirectory(prefix='correction-catalog-') as temporary:
                with patch.object(runner, 'RUNS_DIR', Path(temporary)), \
                     patch.object(runner, 'save_intermediate_value', side_effect=capture):
                    try:
                        result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-catalog-correction',
                            domain + ('-invalid' if invalid else '')).execute(selected, cdg=graph))
                    except RuntimeError as error:
                        if not invalid or 'aligned probabilities in [0,1] required' not in str(error) or \
                                'apply_conditional_offsets' not in str(error):
                            raise
                        result = dict(status='rejected_by_atom')
            if invalid:
                require(result['status'] == 'rejected_by_atom' and 'output' not in captured, 'Invalid input not rejected')
            else:
                require(result['status'] == 'completed', 'Catalog graph execution failed')
                np.testing.assert_allclose(captured['output'], predictions + scale*np.array([3., -2., 0.]), atol=1e-13)
            cases.append(dict(domain=domain, invalid_probability=invalid, status=result['status']))
    return dict(format='conditional-correction-catalog-execution.v1', passed=True, synthetic_only=True,
        version_id=proposed['version_id'], graph_sha256=proposed['graph_sha256'],
        exact_stored_bindings_verified=True, scenarios=cases, validator_sha256=sha(Path(__file__)),
        catalog_mutations=0, approved=False,
        limitation='Explicit draft retrieval and execution; served selection and empirical domain effectiveness are not established.')


if __name__ == '__main__':
    report = validate()
    (ROOT / 'docs/reviews/conditional_correction_catalog_execution.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
