"""Execute the complete temporal sparse lifecycle through serialized CDG nodes."""
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from temporal_sparse_synthetic import payload
import sciona.atoms.ml.temporal_sparse_execution as provider
from sciona.temporal_sparse_graph import build_temporal_sparse_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner


def main():
    digest, nodes, edges = encode_execution_graph(build_temporal_sparse_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(graph)[0] == digest
    runner._ensure_atoms_imported()
    captured = {}
    def capture(directory, node, name, value):
        if node == 'execute' and name == 'out_result':
            captured['result'] = value
    with tempfile.TemporaryDirectory(prefix='temporal-sparse-synthetic-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), patch.object(runner, 'save_intermediate_value', side_effect=capture):
            result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-temporal-sparse', 'case').execute(
                {'payload': payload(Path(temporary))}, cdg=graph))
    assert result['status'] == 'completed', result['status']
    values = json.loads(json.dumps(captured['result'], allow_nan=False))
    assert (values['training_rows'], values['validation']['rows'], values['query_rows']) == (30, 10, 2)
    assert values['models'] == 2 and values['updates_per_model'] == 5 and values['feature_count'] == 24
    scores = np.asarray(values['predictions'])
    assert scores.shape == (2,) and np.isfinite(scores).all() and ((scores >= 0) & (scores <= 20)).all()
    assert np.isfinite(values['validation']['rmsle']) and values['validation']['rmsle'] >= 0
    assert provider.witness_temporal_sparse_prepare({}) == {'kind': 'TemporalSparse.Prepared'}
    assert provider.witness_temporal_sparse_execute({'kind': 'TemporalSparse.Prepared'}) == {'kind': 'TemporalSparse.Result'}
    paths = [str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('temporal_sparse_*.py'))] + ['scripts/temporal_sparse_synthetic.py', 'scripts/validate_temporal_sparse_graph_execution.py']
    report = dict(status='passed', approved=False, serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2, training_rows=30, validation_rows=10, query_rows=2,
                    models=2, updates_per_model=5, feature_count=24, bounded_predictions=True,
                    finite_rmsle=True, strict_json_output=True, graph_codec_roundtrip=True, provider_witness_contracts=True),
        scope='Synthetic six-stage private-stream temporal sparse execution; no served publication or historical accuracy claim.')
    (ROOT/'docs/reviews/competition_temporal_sparse_graph_execution.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__':
    main()
