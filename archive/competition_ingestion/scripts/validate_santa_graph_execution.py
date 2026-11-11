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
from santa_synthetic import payload
import sciona.atoms.ml.santa_execution as provider
from sciona.santa_graph import build_santa_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner


def main():
    digest, nodes, edges = encode_execution_graph(build_santa_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(graph)[0] == digest
    runner._ensure_atoms_imported()
    captured = {}
    def capture(directory, node, name, value):
        if node == 'execute' and name == 'out_result':
            captured['result'] = value
    with tempfile.TemporaryDirectory(prefix='santa-synthetic-') as temporary:
        with patch.object(runner, 'RUNS_DIR', Path(temporary)), patch.object(runner, 'save_intermediate_value', side_effect=capture):
            result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-santa', 'case').execute(
                {'payload': payload()}, cdg=graph))
    assert result['status'] == 'completed', result['status']
    values = json.loads(json.dumps(captured['result'], allow_nan=False))
    assert values['training_rows'] > 0 and values['validation_rows'] > 0
    assert values['models'] == 2 and values['query_step'] == 1
    scores = np.asarray(values['scores'])
    assert scores.shape == (100,) and np.isfinite(scores).all()
    assert 0 <= values['action'] < 100
    assert all(1 <= e['best_iteration'] <= 24 and np.isfinite(e['heldout_rmse']) for e in values['model_evaluation'].values())
    assert provider.witness_santa_prepare({}) == {'kind': 'Santa.Prepared'}
    assert provider.witness_santa_execute({'kind': 'Santa.Prepared'}) == {'kind': 'Santa.Result'}
    paths = [str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('santa_*.py'))] + ['scripts/santa_synthetic.py', 'scripts/validate_santa_graph_execution.py']
    report = dict(status='passed', approved=False, serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=2, training_rows=values['training_rows'], validation_rows=values['validation_rows'],
                    models=2, query_step=1, action_count=100, finite_scores=True, heldout_best_iterations=True,
                    strict_json_output=True, graph_codec_roundtrip=True, provider_witness_contracts=True),
        scope='Synthetic corrected Santa replay-to-action execution; no served publication or historical performance claim.')
    (ROOT/'docs/reviews/competition_santa_graph_execution.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__':
    main()
