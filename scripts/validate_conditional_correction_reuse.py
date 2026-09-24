"""Source comparison and actual cross-domain synthetic CDG execution."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

import sciona.atoms.ml.calibration.conditional_residuals as provider
from sciona.conditional_correction_graph import build_conditional_correction_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner


def main(source, output):
    hashes = {'Train_Models.source.py': '2f8c7b7eb6c6c174128c37bc7d53c1db3236a9320699017f02387b406964fc03',
              'Run_Inference.source.py': '5d78b1dd59c14597deac54ce482ef08150eb6cc4dad99166cc8dfd7a49374c44'}
    texts = {}
    for name, digest in hashes.items():
        raw = (source/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('pinned source code required')
        texts[name] = raw.decode().splitlines()
    calibration = [line.strip() for line in texts['Train_Models.source.py'] if line.strip().startswith(('median_underestimation = (', 'median_overestimation = ('))]
    application = [line.strip() for line in texts['Run_Inference.source.py'] if line.strip().startswith("X_test_lower['final_pred_minutes_until_pushback'] = np.where")]
    assert len(calibration) == len(application) == 2
    random = np.random.default_rng(2131)
    for _ in range(64):
        prediction, target = random.normal(size=(2, 65))
        probability = random.random(65)
        probability[:3] = [.1, .5, .9]
        frame = pd.DataFrame(dict(actual_minutes_until_pushback=target, pred_minutes_until_pushback=prediction))
        namespace = dict(np=np, df_test_estimate_underestimate=frame[probability>.5],
                         df_test_estimate_overestimate=frame[probability<.5])
        exec('\n'.join(calibration), namespace)
        offsets = provider.estimate_conditional_offsets(target, prediction, probability, .5)
        np.testing.assert_array_equal(offsets, [namespace['median_underestimation'], namespace['median_overestimation']])
        namespace.update(X_test_lower=frame.copy(), y_prob_estimate=probability)
        exec('\n'.join(application), namespace)
        np.testing.assert_array_equal(provider.apply_conditional_offsets(prediction, probability, offsets, .5),
                                      namespace['X_test_lower']['final_pred_minutes_until_pushback'].to_numpy())

    digest, nodes, edges = encode_execution_graph(build_conditional_correction_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(graph)[0] == digest
    runner._ensure_atoms_imported()
    cases = []
    for domain, base, scale in [('manufacturing_dimensions', 2., .01), ('energy_forecasts', 500., 20.)]:
        calibration_prediction = base+scale*np.array([1., 2., 3., 4., 5.])
        prediction = base+scale*np.array([8., 9., 10.])
        payload = dict(observed=calibration_prediction+scale*np.array([2., 4., -1., -3., 999.]),
            calibration_predictions=calibration_prediction, calibration_probabilities=np.array([.8, .9, .1, .2, .5]),
            predictions=prediction, probabilities=np.array([.7, .3, .5]), threshold=.5)
        captured = {}
        def capture(directory, node, name, value):
            if node == 'correct':
                captured['output'] = value
        with tempfile.TemporaryDirectory(prefix='conditional-correction-') as temporary:
            with patch.object(runner, 'RUNS_DIR', Path(temporary)), patch.object(runner, 'save_intermediate_value', side_effect=capture):
                result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-correction', domain).execute(payload, cdg=graph))
        assert result['status'] == 'completed', result['status']
        np.testing.assert_allclose(captured['output'], prediction+scale*np.array([3., -2., 0.]), atol=1e-13)
        cases.append(dict(domain=domain, calibration_examples=5, application_examples=3, actual_runner_completed=True))
    root = Path(__file__).resolve().parents[1]
    files = ['sciona/conditional_correction_graph.py', 'scripts/validate_conditional_correction_reuse.py', 'tests/test_conditional_correction.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
                  source_float64_cases=64, source_sha256=hashes, serialized_graph_sha256=digest,
                  reusable_nodes=2, explicit_graph_inputs=6, scenarios=cases,
                  implementation_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
                  provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                  limits=['Float64 numeric source comparison; original estimator float32 and submission rounding remain separate.',
                          'Constructed probabilities and calibration residuals, not trained classifiers or empirical domain validation.',
                          'Calibration population quality and target units must be established by the caller.',
                          'Complete NASA pipelines and publication qualification remain pending.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_directory, args.output)
