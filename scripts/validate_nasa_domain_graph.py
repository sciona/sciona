"""Run explicit domain nodes against the existing ten-slot pinned source oracle."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import sciona.atoms.ml.domain_adapters.airport_features as provider
from sciona.nasa_domain_graph import build_nasa_domain_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.visualizer import runner
import scripts.validate_residual_classifier_graph as oracle

ROOT = Path(__file__).resolve().parents[1]


def validate(source):
    bound_path = ROOT/'docs/reviews/residual_classifier_bound_execution.json'
    bound = json.loads(bound_path.read_text())
    paths = {}
    for name, expected in bound['execution_source_sha256'].items():
        path = Path(importlib.import_module(name).__file__).resolve() if name.startswith('sciona.atoms.ml.') else ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Previously qualified execution dependency drift: '+name)
        paths[name] = path
    paths.update(domain_provider=Path(provider.__file__), domain_builder=ROOT/'sciona/nasa_domain_graph.py',
                 domain_validator=Path(__file__), materialized_preflight=ROOT/'sciona/residual_metadata_preflight.py')
    pins = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    graph = build_nasa_domain_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    current, attached = {}, []
    original_tables = oracle.source_level_tables
    original_execute = runner.CDGExecutionSession.execute
    controls = ['train_fraction','seed','maximum_error','threshold','prediction_feature_name']
    def tables(rng, airport, training, vocabulary=None):
        result = original_tables(rng, airport, training, vocabulary)
        current['airport'] = airport
        current['training_records' if training else 'prediction_records'] = result[2]
        return result
    async def execute(session, user_inputs, **kwargs):
        # Only raw runtime records and controls enter the graph. In particular,
        # no precomputed numerical matrices can mask a missing domain edge.
        payload = {name:user_inputs[name] for name in controls}
        payload.update(current)
        old_capture = runner.save_intermediate_value
        captured = {}
        def capture(directory, node, name, value):
            if node == 'domain_handoff' and name == 'out_features':
                captured['handoff_completed'] = True
            if node == 'internal_fit' and name == 'out_state' and not captured.get('handoff_completed'):
                raise ValueError('Model fit preceded materialized metadata preflight')
            if node == 'domain_output' and name == 'out_prediction_table':
                captured['output'] = value
            if node == 'final' and name == 'out_predictions':
                captured['numeric'] = value
            old_capture(directory, node, name, value)
        with patch.object(runner, 'save_intermediate_value', side_effect=capture):
            outcome = await original_execute(session, payload, **kwargs)
        if outcome['status'] != 'completed' or 'output' not in captured or not captured.get('handoff_completed'):
            raise ValueError('Complete domain graph did not execute')
        expected = current['prediction_records']['queries'][['gufi','timestamp','airport']].copy().reset_index(drop=True)
        expected['minutes_until_pushback'] = captured['numeric']
        pd.testing.assert_frame_equal(captured['output'], expected)
        attached.append(len(expected))
        return outcome
    with patch.object(oracle, 'build_residual_classifier_graph', return_value=graph), \
            patch.object(oracle, 'source_level_tables', side_effect=tables), \
            patch.object(runner.CDGExecutionSession, 'execute', execute):
        result = oracle.validate(source)
    if result['graph_sha256'] != digest or len(attached) != 10:
        raise ValueError('Domain graph evidence mismatch')
    if pins != {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}:
        raise ValueError('Execution dependency changed during validation')
    try:
        provider.witness_unmaterialized()
    except ValueError:
        symbolic_rejected = True
    else:
        raise ValueError('Unqualified symbolic propagation accepted')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        graph_sha256=digest, nodes=len(nodes), edges=len(edges), domain_nodes=15,
        source_scenarios=result['scenarios'], output_rows=sum(attached),
        raw_records_only_graph_inputs=True, attached_identity_order_exact=True,
        materialized_numerical_preflight_before_fitting=True,
        whole_graph_symbolic_qualified=False, unsupported_symbolic_request_rejected=symbolic_rejected,
        execution_source_sha256=pins,
        numerical_bound_evidence_sha256=hashlib.sha256(bound_path.read_bytes()).hexdigest(),
        implementation_sha256={
            'provider':hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
            'builder':hashlib.sha256((ROOT/'sciona/nasa_domain_graph.py').read_bytes()).hexdigest(),
            'source_oracle':hashlib.sha256(Path(oracle.__file__).read_bytes()).hexdigest()},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Actual domain branches are graph nodes; provider metadata and whole-graph symbolic table propagation remain unqualified.',
                    'Single-airport graph only; dispatcher graph composition and persistent training lifecycle remain pending.',
                    'Explicitly corrected source behavior, not original bug-for-bug reproduction or empirical competition performance.',
                    'No catalog import or approval of the new domain adapters is implied.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_directory)
    (ROOT/'docs/reviews/competition_nasa_domain_graph.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='source_scenarios'}))
