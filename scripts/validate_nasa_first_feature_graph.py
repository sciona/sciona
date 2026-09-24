"""Compare decomposed production execution against the qualified raw adapter."""
import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import tempfile
from unittest.mock import patch

import pandas as pd

from sciona.nasa_first_feature_graph import build_nasa_first_feature_graph
from sciona.nasa_first_feature_adapter import nasa_first_feature_tables
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.synthetic_nasa_first_workflow_inputs import population_inputs
from scripts.validate_nasa_first_model_handoff import ROOT, sha


def validate():
    original = build_nasa_first_feature_graph()
    digest, nodes, edges = encode_execution_graph(original)
    graph = decode_execution_graph(nodes, edges, digest)
    if graph != original:
        raise ValueError('Feature graph roundtrip differs')
    outcomes = []
    with tempfile.TemporaryDirectory(prefix='sciona-first-feature-graph-') as directory:
        def execute(queries, raw, *, expected_error=None):
            before = queries.copy(deep=True)
            captured = {}
            def capture(path, node, port, value):
                if node == 'assemble' and port.startswith('out_'):
                    captured['result'] = value
            with patch.object(runner, 'RUNS_DIR', Path(directory)), patch.object(runner, 'save_intermediate_value', side_effect=capture), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-first-features', str(len(outcomes))).execute(
                        dict(queries=queries, raw_options=raw), cdg=graph))
                except RuntimeError as error:
                    cause = error.__context__
                    if expected_error is None or not isinstance(cause, ValueError) or str(cause) != expected_error:
                        raise
                    outcomes.append(dict(expected_failure=True))
                    return
            if expected_error is not None or result['status'] != 'completed' or 'result' not in captured:
                raise ValueError('Feature graph execution differs')
            expected = nasa_first_feature_tables(queries, **raw)
            pd.testing.assert_frame_equal(captured['result'], expected)
            pd.testing.assert_frame_equal(queries, before)
            outcomes.append(dict(expected_failure=False, rows=len(queries), columns=len(expected.columns)))
            return captured['result']

        queries, raw = runpy.run_path(str(ROOT/'tests/test_nasa_first_feature_adapter.py'))['fixture']()
        execute(queries, raw)
        execute(queries.iloc[[2, 0, 2]], raw)
        bad = copy.copy(raw)
        bad['entity_snapshot_available_at'] = queries['timestamp'].max()+pd.Timedelta(days=1)
        execute(queries, bad, expected_error='Queries must lie in context with an available entity snapshot')
        for population in range(10):
            _, _, queries, raw, _ = population_inputs(population, pd.Timestamp('2030-01-01'))
            execute(queries, raw)
    provider = ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/first_place_features.py'
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        graph_sha256=digest, nodes=len(graph.nodes), edges=len(graph.edges),
        reused_generic_operations=9, new_domain_adapters=2, codec_roundtrip=True, production_executor=True,
        outcomes=outcomes, input_queries_unchanged=True, domain_provider_sha256=sha(provider),
        implementation_sha256={name: sha(ROOT/name) for name in ['sciona/nasa_first_graph_adapters.py',
            'sciona/nasa_first_feature_graph.py', 'sciona/nasa_first_feature_adapter.py',
            'scripts/validate_nasa_first_feature_graph.py', 'scripts/synthetic_nasa_first_workflow_inputs.py',
            'sciona/available_feature_contracts.py', 'sciona/visualizer/runner.py', 'sciona/services/execution_graph_codec.py']},
        limitations=['All comparisons use independent synthetic fixtures and the previously qualified corrected adapter.',
            'Output persistence is replaced by capture; graph dispatch and every feature operation use the production executor.',
            'Full model training, imputation, prediction and fallback graph composition remains pending.',
            'Domain adapters are locally registered; their catalog publication and exact version binding remain pending.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/competition_nasa_first_decomposed_feature_graph.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, nodes=report['nodes'], edges=report['edges'],
        successful_cases=sum(not case['expected_failure'] for case in report['outcomes']))))
