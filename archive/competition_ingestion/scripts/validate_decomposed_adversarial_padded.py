"""All complete padded neural schedules with synthetic caller-supplied states."""
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
from sciona import decomposed_adversarial_lifecycle as lifecycle
from sciona.decomposed_adversarial_rgb_attack import attack_rgb
from sciona.decomposed_adversarial_updates import update
from sciona.adversarial_updates import update as reference_update
from sciona.adversarial_ensemble import ORDER
from sciona.adversarial_inception_v3 import build_inception_v3
from sciona.adversarial_inception_v4 import build_inception_v4
from sciona.adversarial_inception_resnet import build_inception_resnet
from sciona.adversarial_resnet import build_resnet101
from sciona.adversarial_state_mapping import mapping
from sciona.adversarial_image_boundary import source_rgb
import sciona.atoms.numerical.statistics.row_scale as provider


def synthetic_states(branch):
    specs = {}
    for i, scope in enumerate(ORDER[branch]):
        if scope == 'resnet_v2_101':
            builder, family = build_resnet101, 'resnet101'
        elif scope == 'InceptionV4':
            builder, family = build_inception_v4, 'inception_v4'
        elif 'Resnet' in scope:
            builder, family = build_inception_resnet, 'inception_resnet'
        else:
            builder, family = build_inception_v3, 'inception_v3'
        model = builder(seed=1400+i)
        # Synthetic nonzero offsets keep padded-zero derivatives defined;
        # no epsilon stabilizer or substituted normalization is introduced.
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                if name.endswith('.beta'):
                    buffer.fill_(.1)
        state = model.state_dict()
        source = {}
        for key, (name, layout) in mapping(model, family, scope).items():
            value = state[key]
            source[name] = (value.permute(2, 3, 1, 0) if layout == 'HWIO'
                            else value.T if layout == 'IO' else value)
        specs[scope] = {'kind': 'slim_tensors', 'seed': 1400+i, 'tensors': source}
        del model, state
    return specs


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checks = {}
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
        rng = torch.get_rng_state().clone()

        def observe(x, gradient, noise, lo, hi, **kwargs):
            nonlocal observed, previous_x, previous_noise
            np.testing.assert_array_equal(x, previous_x)
            np.testing.assert_array_equal(noise, previous_noise)
            np.testing.assert_array_equal(lo, lower)
            np.testing.assert_array_equal(hi, upper)
            assert gradient.shape == initial.shape
            expected = reference_update(x, gradient, noise, lo, hi, **kwargs)
            actual = update(x, gradient, noise, lo, hi, **kwargs)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            previous_x, previous_noise = (a.copy() for a in actual)
            observed += 1
            if observed % 5 == 0:
                print(json.dumps({'branch': branch, 'verified_steps': observed}), flush=True)
            return actual

        with patch.object(lifecycle, 'update', side_effect=observe):
            result = attack_rgb(rgb, mode=mode, epsilon=epsilon, initializations=specs,
                                targets=np.array([17], np.int64) if mode == 'targeted' else None)
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
        checks[branch] = dict(full_models=len(specs), complete_iterations=steps,
                              exact_reference_update_checks=observed, neural_batch_size=10,
                              real_entries=1, padded_entries=9, only_real_entry_encoded=True,
                              decoded_png_matches_source_scaling=True,
                              fixed_projection_bounds=True, input_and_rng_unchanged=True)
        print(json.dumps({branch: checks[branch]}), flush=True)
        del result, specs
        gc.collect()
    paths = ['sciona/decomposed_adversarial_rgb_attack.py', 'sciona/decomposed_adversarial_lifecycle.py',
             'sciona/decomposed_adversarial_updates.py', 'scripts/validate_decomposed_adversarial_padded.py',
             'sciona/adversarial_image_boundary.py', 'sciona/adversarial_ensemble.py',
             'sciona/adversarial_state_mapping.py', 'sciona/adversarial_updates.py',
             'docs/reviews/competition_adversarial_semantic_review.json']
    report = dict(format='decomposed-adversarial-padded-validation.v1', result='passed',
                  approved=False, catalog_mutations=0, checks=checks,
                  generic_provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
                  sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  limits='Synthetic caller states with BN beta=.1; no pretrained quality or historical TensorFlow binary equivalence. Exact update comparison uses the approved CPU implementation on every actual neural gradient. Source PNG min/max stretching does not guarantee a pixel epsilon bound. Explicit decomposed graph execution and catalog publication remain pending.')
    (ROOT/'docs/reviews/decomposed_adversarial_padded.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
