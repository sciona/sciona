"""Synthetic source-parity checks; no competition inputs or weights required."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sciona.contrails_preprocessing import SpatialTTA, ash_color


def validate(root, source_root):
    pins = json.loads((root / 'docs/reviews/competition_contrails_source_pins.json').read_text())
    for item in pins['files']:
        if hashlib.sha256((source_root / item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Public source pin mismatch')
    # Compile only reviewed pure functions from the hash-verified source.
    namespace = {'torch': torch}
    data_tree = ast.parse((source_root / 'src/unet5/data.py').read_text())
    selected = [n for n in data_tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('rescale_range', 'ash_color')]
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned-color-functions>', 'exec'), namespace)
    tta_namespace = {}
    exec(compile((source_root / 'src/submit/unet5/tta.py').read_text(), '<pinned-spatial-tta>', 'exec'), tta_namespace)
    rng = np.random.default_rng(41203)
    counts = {'color_source_cases': 0, 'tta_source_cases': 0, 'tta_independent_oracles': 0,
              'unclipped_output_case': 0, 'invalid_inputs_rejected': 0}
    for dtype in (torch.float32, torch.float64):
        for side in (3, 8):
            x = torch.tensor(rng.uniform(200, 350, (2, 3, side, side)), dtype=dtype)
            expected = torch.stack([namespace['ash_color'](sample) for sample in x])
            torch.testing.assert_close(ash_color(x), expected, rtol=0, atol=0)
            counts['color_source_cases'] += 1
            for batch in (1, 3):
                images = torch.tensor(rng.normal(size=(batch, 2, side, side)), dtype=dtype)
                for mode in SpatialTTA._modes:
                    ours, source = SpatialTTA(mode), tta_namespace['TTA'](mode)
                    aug = ours.stack(images)
                    torch.testing.assert_close(aug, source.stack(images), rtol=0, atol=0)
                    # Deliberately orientation-dependent predictions expose inverse-transform bugs.
                    pred = 0.1 * aug[:, :1].square() + 0.1 * torch.arange(side, dtype=dtype).reshape(1, 1, 1, side)
                    actual = ours.average(pred)
                    torch.testing.assert_close(actual, source.average(pred), rtol=0, atol=0)
                    counts['tta_source_cases'] += 1
                    values = pred.numpy().reshape(ours.n, batch, 1, side, side)
                    if ours.prob:
                        values = 1 / (1 + np.exp(-values))
                    restored = []
                    for i, value in enumerate(values):
                        k = i // 2 if ours.n == 8 else i
                        if ours.n == 8 and i % 2:
                            value = np.flip(value, axis=-1)
                        restored.append(np.rot90(value, -k, axes=(-2, -1)))
                    expected_np = np.mean(restored, axis=0)
                    if ours.prob:
                        expected_np = np.clip(expected_np, 1e-6, 1 - 1e-6)
                        expected_np = np.log(expected_np / (1 - expected_np))
                    np.testing.assert_allclose(actual.numpy(), expected_np, rtol=2e-4, atol=2e-4)
                    counts['tta_independent_oracles'] += 1
    extreme = torch.tensor([[[200.]], [[350.]], [[400.]]])
    assert torch.any(ash_color(extreme) < 0)
    counts['unclipped_output_case'] = 1
    # Source clips in float32: its upper bound differs from exact 1 - 1e-6.
    for value in (-100., 100.):
        saturated = torch.full((8, 1, 2, 2), value, dtype=torch.float64)
        result = SpatialTTA('d4prob').average(saturated)
        boundary = np.float32(1e-6 if value < 0 else 1 - 1e-6)
        expected = np.log(boundary / (np.float32(1) - boundary))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)
    counts['float32_saturation_oracles'] = 2
    invalid = [lambda: ash_color(torch.zeros(2, 3, 3)),
               lambda: ash_color(torch.full((3, 2, 2), float('nan'))),
               lambda: SpatialTTA('bad'),
               lambda: SpatialTTA('d4prob').stack(torch.zeros(1, 1, 2, 3)),
               lambda: SpatialTTA('rotprob').average(torch.zeros(3, 1, 2, 2)),
               lambda: SpatialTTA('none').average(torch.zeros(1, 2, 2, 2))]
    for call in invalid:
        try:
            call()
        except ValueError:
            counts['invalid_inputs_rejected'] += 1
        else:
            raise AssertionError('Invalid input accepted')
    return {'approved': False, 'checks': counts, 'source_commit': pins['commit'],
            'implementation_sha256': hashlib.sha256((root / 'sciona/contrails_preprocessing.py').read_bytes()).hexdigest(),
            'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'Synthetic preprocessing and spatial TTA only. Full temporal models, training and ensemble still pending.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate(root, args.source_root)
    (root / 'docs/reviews/competition_contrails_preprocessing.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['checks']))
