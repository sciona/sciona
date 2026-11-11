"""Exercise both full sweeps and optional precedence through the actual runner."""
import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sciona.atoms.ml.connectomics_execution as provider
from sciona.connectomics_graph import build_connectomics_graph
from sciona.connectomics_source import source_namespace
from sciona.connectomics_runtime import prepare, execute
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner


def reference_directivity(x):
    filtered = np.zeros(x.shape)
    for t in range(1, len(x)-1):
        filtered[t] = x[t] + x[t-1] + .8*x[t-2] + .4*x[t-3]
    diff = np.diff(filtered, axis=0)
    diff[diff < .12] = 0
    diff = diff ** .9
    counts = np.zeros((x.shape[1], x.shape[1]))
    for t in range(len(diff)-1):
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                if i != j and diff[t,i]+.2 < diff[t+1,j] < diff[t,i]+.5:
                    counts[i,j] += 1
    result = counts-counts.T
    np.fill_diagonal(result, result.min())
    return (result-result.min()) / np.ptp(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    x = np.asfortranarray(np.random.default_rng(1729).uniform(0,.8,(240,10)), dtype=np.float32)
    original = x.copy()
    ns = source_namespace(args.source)
    ns['print'] = lambda *a, **kw: None
    directed = reference_directivity(x)
    np.testing.assert_array_equal(directed, ns['make_prediction_directivity'](x, n_jobs=1))
    digest, nodes, edges = encode_execution_graph(build_connectomics_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(graph)[0] == digest
    runner._ensure_atoms_imported()
    cases = []
    for mode in ('simple', 'tuned'):
        expected = ns['make_'+mode+'_inference'](x)
        for directivity in (False, True):
            captured = {}
            def capture(directory, node, name, value):
                if node == 'execute' and name == 'out_result':
                    captured['result'] = value
            with tempfile.TemporaryDirectory(prefix='connectomics-synthetic-graph-') as temporary:
                with patch.dict(os.environ, {'SCIONA_CONNECTOMICS_SOURCE_DIR':str(args.source)}), \
                     patch.object(runner, 'RUNS_DIR', Path(temporary)), \
                     patch.object(runner, 'save_intermediate_value', side_effect=capture):
                    status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-connectomics', 'case').execute(
                        {'payload':dict(version=1, signals=x.tolist(), mode=mode, directivity=directivity)}, cdg=graph))
            assert status['status'] == 'completed', status['status']
            result = json.loads(json.dumps(captured['result'], allow_nan=False))
            target = .997*expected+.003*directed if directivity else expected
            np.testing.assert_array_equal(result['scores'], target)
            assert result['pca_fits'] == (240 if mode=='simple' else 480)
            assert result['mode']==mode and result['directivity']==directivity
            cases.append(dict(mode=mode, directivity=directivity, pca_fits=result['pca_fits'], exact_source_match=True))
    np.testing.assert_array_equal(x, original)
    with patch.dict(os.environ, {'SCIONA_CONNECTOMICS_SOURCE_DIR':str(args.source)}):
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                execute(prepare(dict(version=1, signals=x*1e-4, mode='simple', directivity=False)))
            except ValueError as error:
                assert str(error)=='Degenerate residual covariance'
            else:
                raise AssertionError('Degenerate covariance accepted')
    with tempfile.TemporaryDirectory(prefix='connectomics-synthetic-tamper-') as temporary:
        cache=Path(temporary)/'source'
        shutil.copytree(args.source, cache)
        (cache/'code/PCA.py').write_bytes(b'changed source')
        try:
            source_namespace(cache)
        except ValueError as error:
            assert 'hash mismatch' in str(error)
        else:
            raise AssertionError('Tampered source accepted')
    assert provider.witness_connectomics_prepare({}) == {'kind':'Connectomics.Prepared'}
    assert provider.witness_connectomics_execute({'kind':'Connectomics.Prepared'}) == {'kind':'Connectomics.Result'}
    paths = [str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('connectomics_*.py'))]
    paths.append('scripts/validate_connectomics_graph_execution.py')
    report = dict(format='connectomics-graph-execution.v1', status='passed', approved=False,
        serialized_graph_sha256=digest, cases=cases, independent_directivity_exact=True,
        actual_runner_nodes=2, strict_json_output=True, graph_codec_roundtrip=True,
        provider_witness_contracts=True, caller_input_preserved=True,
        degenerate_covariance_rejected=True, tampered_source_rejected=True,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='Synthetic execution only; no historical accuracy or served publication claim. Private matrices remain in memory or temporary storage.')
    (ROOT/'docs/reviews/competition_connectomics_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',cases=cases)))


if __name__ == '__main__':
    main()
