"""Execute the serialized model handoff with real qualified synthetic models."""
import argparse
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from sciona.model_slot_graph import build_model_slot_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.validate_nasa_first_model_handoff import ROOT, sha, validate as validate_native


def validate(source, checkpoint):
    original = build_model_slot_graph()
    digest, nodes, edges = encode_execution_graph(original)
    graph = decode_execution_graph(nodes, edges, digest)
    if graph != original:
        raise ValueError('Model handoff graph codec differs')
    runs = []
    with tempfile.TemporaryDirectory(prefix='sciona-model-handoff-') as directory:
        def execute(models, population_bindings, shared_bindings, population, *, expected_error=None):
            captured = {}
            def capture(path, node, port, value):
                if node == 'select' and port == 'out_slots':
                    captured['slots'] = value
            with patch.object(runner, 'RUNS_DIR', Path(directory)), patch.object(runner, 'save_intermediate_value', side_effect=capture), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-model-handoff', str(len(runs))).execute(
                        dict(models=models, population_bindings=population_bindings, shared_bindings=shared_bindings,
                             population=population), cdg=graph))
                except RuntimeError as error:
                    cause = error.__context__
                    if expected_error is None or type(cause) is not expected_error[0] or str(cause) != expected_error[1]:
                        raise
                    result = {'status': 'failed'}
            passed = result['status'] == 'completed'
            if expected_error is not None:
                if passed or captured:
                    raise ValueError('Invalid model handoff executed')
            elif not passed or 'slots' not in captured:
                raise ValueError('Native graph handoff failed')
            runs.append(dict(expected_failure=expected_error is not None, completed=passed))
            return captured.get('slots')

        native = validate_native(source, checkpoint, graph_selector=execute)
        # Reuse with a different application vocabulary and opaque runtime model.
        model = object()
        selected = execute({'regressor': model}, {'manufacturing-cell': {'estimate': 'regressor'}}, {}, 'manufacturing-cell')
        if selected != {'estimate': model}:
            raise ValueError('Cross-domain handoff differs')
        for models, bindings, shared, population, error in [
            ({'m': model}, {'p': {'x': 'missing'}}, {}, 'p', (ValueError, 'Binding references a missing model')),
            ({'m': model}, {'p': {'x': 'm'}}, {'x': 'm'}, 'p', (ValueError, 'Local and shared slots overlap')),
            ({'m': model}, {'p': {'x': 'm'}}, {}, 'unknown', (KeyError, "'unknown'")),
            ({'m': model, 'unused': object()}, {'p': {'x': 'm'}}, {}, 'p', (ValueError, 'Unreferenced model state')),
        ]:
            execute(models, bindings, shared, population, expected_error=error)
    from sciona.ghost.registry import REGISTRY
    for node in graph.nodes:
        if node.matched_primitive not in REGISTRY:
            raise ValueError('Standard loader did not register model operation')
    provider = ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/model_selection/model_banks.py'
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True, native_threads=1,
        graph_sha256=digest, nodes=len(graph.nodes), edges=len(graph.edges), codec_roundtrip=True,
        production_executor=True, standard_loader_discovery=True, native_handoff=native,
        successful_graphs=sum(item['completed'] for item in runs), rejected_graphs=sum(item['expected_failure'] for item in runs),
        provider_source_sha256=sha(provider),
        implementation_sha256={name: sha(ROOT/name) for name in ['sciona/model_slot_graph.py', 'sciona/model_slot_bank.py',
            'scripts/validate_model_slot_graph.py', 'scripts/validate_nasa_first_model_handoff.py',
            'sciona/visualizer/runner.py', 'sciona/services/execution_graph_codec.py']},
        limitations=['Only graph topology is serialized; native models remain private in-process runtime objects.',
            'Intermediate persistence is replaced by output capture; provider dispatch and graph edges use the production executor.',
            'This is the model-state handoff subgraph, not the complete first-place training/inference CDG.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_directory, args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_model_slot_graph.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, successful_graphs=report['successful_graphs'], rejected_graphs=report['rejected_graphs'],
        native_query_comparisons=report['native_handoff']['distinct_query_comparisons'])))
