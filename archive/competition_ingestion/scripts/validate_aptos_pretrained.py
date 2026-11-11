"""Check pinned mirror weights on synthetic inputs; no competition data."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
from safetensors.torch import load_file

from sciona.aptos_initialization import build_from_checkpoint
from sciona.aptos_models import FAMILIES
from sciona.aptos_preprocessing import prepare_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', required=True, type=Path)
    parser.add_argument('--family', required=True, choices=FAMILIES)
    args = parser.parse_args()
    family = args.family
    pins_path = Path('docs/reviews/competition_aptos_pretrained_mirror_pins.json')
    pins = json.loads(pins_path.read_text())['artifacts'][family]
    checkpoint = args.artifact_root / (family + '.safetensors')
    torch.set_num_threads(2)
    torch.manual_seed(719)
    with patch('torch.utils.model_zoo.load_url', side_effect=AssertionError('Implicit download')):
        model = build_from_checkpoint(family, checkpoint,
            expected_sha256=pins['sha256'], checkpoint_format='safetensors')
    original = load_file(checkpoint)
    state = model.state_dict()
    checked = 0
    for key, value in original.items():
        if key.startswith('last_linear.'):
            continue
        assert key in state
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        checked += 1
    _, pool, _, size = FAMILIES[family]
    absent = set(state) - set(original) - {pool + '.p'}
    # Old publisher checkpoints predate torch BatchNorm's batch counter.
    assert all(k.endswith('.num_batches_tracked') and state[k].item() == 0 for k in absent)
    del state, original
    gc.collect()
    model.eval()
    synthetic_rgb = (np.arange(31 * 47 * 3).reshape(31, 47, 3) % 256).astype(np.uint8)
    x = prepare_rgb(synthetic_rgb, family).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        features = model.features(x)
        exponent = getattr(model, pool).p
        pooled = features.clamp(min=1e-6).pow(exponent).mean((-2, -1)).pow(1. / exponent)
        expected = torch.nn.functional.linear(pooled, model.last_linear.weight, model.last_linear.bias)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
        assert output.shape == (1, 1) and torch.isfinite(output).all()
    paths = ['sciona/aptos_initialization.py', 'sciona/aptos_models.py',
             'sciona/aptos_pooling.py', 'sciona/aptos_preprocessing.py',
             'scripts/validate_aptos_pretrained.py', str(pins_path)]
    report = {'passed': True, 'approved': False, 'catalog_mutations': 0,
        'synthetic_inputs_only': True, 'pretrained_weights_used': True,
        'family': family, 'artifact_sha256': pins['sha256'],
        'matched_backbone_tensors': checked, 'initialized_legacy_batch_counters': len(absent),
        'full_resolution': size, 'independent_pool_head_comparison': True,
        'scope': 'Pinned maintainer mirror backbone load and full-resolution forward. Scalar head newly initialized. No training, original-winner identity or complete workflow claim.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_aptos_pretrained_' + family + '_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS pretrained mirror full-resolution forward', family, flush=True)


if __name__ == '__main__':
    main()
