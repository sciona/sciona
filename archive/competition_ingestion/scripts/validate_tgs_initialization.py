"""Opt-in public-reference mapping checks; synthetic activations only."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from sciona.tgs_initialization import load_resnet34_reference, load_resnext50_reference
from sciona.tgs_resnext import TGSResNeXt50
from sciona.tgs_torch_models import TGSResNet34


def main(reference_directory, output):
    torch.set_num_threads(2)
    torch.manual_seed(615)
    reports = []
    for variant in (4, 3, 5):
        model = TGSResNet34(variant)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        receipt = load_resnet34_reference(model, reference_directory / 'resnet34',
            '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef')
        for key, value in model.state_dict().items():
            encoder = key.startswith('stem.1.') or (key.startswith('encoder.') and key.split('.')[2] == '0') or (key == 'stem.0.weight' and variant != 5)
            if not encoder:
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert not torch.equal(model.encoder[0][0][0].conv1.weight, before['encoder.0.0.0.conv1.weight'])
        size = 128 if variant == 5 else 256
        model.eval()
        with torch.no_grad():
            result = model(torch.rand(1, 3, size, size))
        assert all(torch.isfinite(v).all() for v in (result if isinstance(result, tuple) else (result,)))
        receipt.update(nonencoder_unchanged=True, synthetic_forward_finite=True)
        reports.append(receipt)
        del before, model
    model = TGSResNeXt50()
    before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(('decoders.', 'prediction.'))}
    receipt = load_resnext50_reference(model, reference_directory / 'resnext50',
        '3bcb9dedb226c5e5cdd3510d25cdc33297f59016a6d7069758024caa13e3172d')
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    # Independent NHWC patch contraction catches swapped spatial/input/output
    # axes and lexicographic (0,1,10,...) group ordering. No full model oracle.
    errors = []
    with h5py.File(reference_directory / 'resnext50', 'r') as source:
        for stage in range(4):
            block = model.stages[stage][0]
            width = block.grouped.in_channels
            x = np.random.default_rng(923 + stage).normal(size=(1, 7, 7, width)).astype(np.float32)
            padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
            stride = block.grouped.stride[0]
            side = (7 + 2 - 3) // stride + 1
            pieces = []
            for group in range(32):
                name = f'stage{stage+1}_unit1_conv2_{group}'
                kernel = source[f'{name}/{name}/kernel:0'][()]
                c = width // 32
                expected = np.empty((1, side, side, c), dtype=np.float32)
                for row in range(side):
                    for col in range(side):
                        patch = padded[:, row*stride:row*stride+3, col*stride:col*stride+3, group*c:(group+1)*c]
                        expected[:, row, col] = np.einsum('nhwc,hwco->no', patch, kernel)
                pieces.append(expected)
            expected = np.concatenate(pieces, axis=-1)
            with torch.no_grad():
                actual = block.grouped(torch.from_numpy(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            errors.append(float(np.max(np.abs(actual - expected))))
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(torch.rand(1, 3, 224, 224) * 255)).all()
    receipt.update(nonencoder_unchanged=True, synthetic_forward_finite=True,
                   independent_group_convolution_max_errors=errors)
    reports.append(receipt)
    files = ['sciona/tgs_initialization.py', 'sciona/tgs_legacy_weights.py',
             'scripts/validate_tgs_initialization.py', 'tests/test_tgs_initialization.py']
    report = dict(passed=True, catalog_mutations=0, reports=reports,
                  scope='Authenticated reference initialization and synthetic checks only; historical runtime equivalence, reuse qualification and full training remain pending.',
                  sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files})
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.reference_directory, args.output)
