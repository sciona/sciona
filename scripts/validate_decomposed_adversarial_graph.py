"""All complete padded neural schedules with synthetic caller-supplied states."""
import asyncio
import base64
import tempfile
import gc
import hashlib
import inspect
from io import BytesIO
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona import decomposed_adversarial_injected as lifecycle
from sciona.decomposed_adversarial_injected import update
from sciona.decomposed_adversarial_graph import build_decomposed_adversarial_graph
from sciona.dfdc_codec import encode_array, encode_state, decode_array
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
import sciona.atoms.dl.adversarial_normalized_execution as execution_provider
from sciona.adversarial_updates import update as reference_update
from sciona.adversarial_ensemble import ORDER
from sciona.adversarial_inception_v3 import build_inception_v3
from sciona.adversarial_inception_v4 import build_inception_v4
from sciona.adversarial_inception_resnet import build_inception_resnet
from sciona.adversarial_resnet import build_resnet101
from sciona.adversarial_state_mapping import mapping
from sciona.adversarial_image_boundary import source_rgb
import sciona.atoms.numerical.statistics.row_scale as provider


from scripts.validate_decomposed_adversarial_padded import synthetic_states


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runner._ensure_atoms_imported()
    checks = {}
    digest, nodes, edges = encode_execution_graph(build_decomposed_adversarial_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    for mode, epsilon, branch, steps in [('non_targeted', 16, 'non_targeted', 10),
                                       ('targeted', 8, 'targeted_large', 20),
                                       ('targeted', 4, 'targeted_small', 40)]:
        specs = synthetic_states(branch)
        rgb = (np.arange(299*299*3, dtype=np.uint32) % 256).astype(np.uint8).reshape(1, 299, 299, 3)
        original = rgb.copy()
        normalized = (rgb.astype(np.float64)/255.*2.-1.).astype(np.float32)
        initial = np.zeros((10, 299, 299, 3), dtype=np.float32)
        initial[:1] = normalized
        lower = np.maximum(initial-2*epsilon/255, -1)
        upper = np.minimum(initial+2*epsilon/255, 1)
        previous_x, previous_noise = initial.copy(), np.zeros_like(initial)
        observed = 0
        normalization_calls = []
        captured = {}
        rng = torch.get_rng_state().clone()

        def observe(x, gradient, noise, lo, hi, **kwargs):
            nonlocal observed, previous_x, previous_noise
            np.testing.assert_array_equal(x, previous_x)
            np.testing.assert_array_equal(noise, previous_noise)
            np.testing.assert_array_equal(lo, lower)
            np.testing.assert_array_equal(hi, upper)
            assert gradient.shape == initial.shape
            assert kwargs['normalizer'] is provider.normalize_rows_by_scale
            expected = reference_update(x, gradient, noise, lo, hi,
                                        **{k:v for k,v in kwargs.items() if k!='normalizer'})
            def profile(frame, event, arg):
                if event == 'call' and frame.f_code is provider.normalize_rows_by_scale.__code__:
                    normalization_calls.append(frame.f_locals['statistic'])
                    assert frame.f_locals['values'].shape == (10, 299*299*3)
            previous_profile = sys.getprofile()
            try:
                sys.setprofile(profile)
                actual = update(x, gradient, noise, lo, hi, **kwargs)
            finally:
                sys.setprofile(previous_profile)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            previous_x, previous_noise = (a.copy() for a in actual)
            observed += 1
            if observed % 5 == 0:
                print(json.dumps({'branch': branch, 'verified_steps': observed}), flush=True)
            return actual

        payload = {'version':1, 'rgb':encode_array(rgb),
                   'targets':encode_array(np.array([17], np.int64)) if mode=='targeted' else None,
                   'config':dict(mode=mode, epsilon=epsilon, momentum=1., non_targeted_iterations=10),
                   'initializations':{scope:{**spec, 'tensors':encode_state(spec['tensors'])}
                                      for scope,spec in specs.items()}}
        def capture(directory, node, name, value):
            if node=='execute' and name=='out_result':captured['result']=value
            if node=='execute' and name=='in_normalizer':
                assert value is provider.normalize_rows_by_scale
                captured['injected_identity_verified']=True
        with tempfile.TemporaryDirectory(prefix='sciona-synthetic-injected-attack-') as directory:
            with patch.object(lifecycle, 'update', side_effect=observe), \
                 patch.object(runner, 'RUNS_DIR', Path(directory)), \
                 patch.object(runner, 'save_intermediate_value', side_effect=capture):
                status=asyncio.run(runner.CDGExecutionSession(None,'synthetic','graph').execute({'payload':payload},cdg=graph))
        assert status['status']=='completed'
        trace={t['node_id']:t for t in status['trace']}
        assert set(trace)=={'prepare','normalize_rows','execute'}
        assert all(not t['cached'] for t in trace.values())
        assert trace['normalize_rows']['callable_reference'] is True
        assert captured['injected_identity_verified']
        expected_statistic='mean_absolute' if mode=='non_targeted' else 'population_std'
        assert normalization_calls==[expected_statistic]*(steps if mode=='non_targeted' else 2*steps)
        encoded=json.loads(json.dumps(captured['result'],allow_nan=False))
        result=dict(images=decode_array(encoded['normalized_images']),labels=decode_array(encoded['labels']),
                    pngs=[base64.b64decode(v,validate=True) for v in encoded['pngs']],batch_losses=encoded['batch_losses'])
        assert encoded['initialization_kinds']=={s:'slim_tensors' for s in specs}
        assert encoded['saved_pixel_bound_guaranteed'] is False
        assert observed == steps == len(result['batch_losses'][0])
        assert len(result['batch_losses']) == len(result['pngs']) == 1
        assert result['images'].shape == rgb.shape and result['labels'].shape == (1,)
        if mode == 'targeted':
            assert result['labels'].tolist() == [17]
        np.testing.assert_array_equal(result['images'], previous_x[:1])
        assert np.max(np.abs(result['images']-normalized)) <= 2*epsilon/255+1e-7
        assert np.any(result['images'] != normalized)
        np.testing.assert_array_equal(np.array(Image.open(BytesIO(result['pngs'][0]))), source_rgb(result['images'][0]))
        np.testing.assert_array_equal(rgb, original)
        assert torch.equal(rng, torch.get_rng_state())
        checks[branch] = dict(actual_runner_nodes=3, registered_callable_identity=True,
                              registered_normalization_calls=len(normalization_calls), full_models=len(specs), complete_iterations=steps,
                              exact_reference_update_checks=observed, neural_batch_size=10,
                              real_entries=1, padded_entries=9, only_real_entry_encoded=True,
                              decoded_png_matches_source_scaling=True,
                              fixed_projection_bounds=True, input_and_rng_unchanged=True)
        print(json.dumps({branch: checks[branch]}), flush=True)
        del result, specs, payload, captured, encoded
        gc.collect()
    paths = ['sciona/decomposed_adversarial_graph.py', 'sciona/decomposed_adversarial_injected.py',
             'sciona/visualizer/runner.py', 'sciona/services/execution_graph_codec.py',
             'sciona/dfdc_codec.py', 'scripts/validate_decomposed_adversarial_graph.py',
             'scripts/validate_decomposed_adversarial_padded.py',
             'sciona/adversarial_image_boundary.py', 'sciona/adversarial_ensemble.py',
             'sciona/adversarial_state_mapping.py', 'sciona/adversarial_updates.py',
             'docs/reviews/competition_adversarial_semantic_review.json']
    report = dict(format='decomposed-adversarial-graph-validation.v1', result='passed',
                  approved=False, catalog_mutations=0, checks=checks, serialized_graph_sha256=digest,
                  execution_provider_sha256=hashlib.sha256(Path(inspect.getfile(execution_provider)).read_bytes()).hexdigest(),
                  generic_provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
                  sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  limits='Synthetic caller states with BN beta=.1; no pretrained quality or historical TensorFlow binary equivalence. Exact update comparison uses the approved CPU implementation on every actual neural gradient. Source PNG min/max stretching does not guarantee a pixel epsilon bound. Actual serialized graph and registered provider calls verified. Catalog publication and served selection remain pending.')
    (ROOT/'docs/reviews/decomposed_adversarial_graph.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
