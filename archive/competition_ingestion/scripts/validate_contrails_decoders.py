"""Compare complete source-sized decoders on synthetic feature pyramids."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from sciona.contrails_single_decoder import UnetDecoder as SingleDecoder
from sciona.contrails_temporal_decoder import UnetDecoder as TemporalDecoder


def validate(root, source_root):
    torch.set_num_threads(2)
    pins = json.loads((root / 'docs/reviews/competition_contrails_source_pins.json').read_text())
    checks = {'forward_cases': 0, 'input_gradient_cases': 0,
              'parameter_gradient_cases': 0, 'checkpoint_and_batchnorm_cases': 0,
              'single_branch_skipped_block_cases': 0}
    for cls, source, channels in [
        (SingleDecoder, 'src/unet1024/unet_decoder.py', [256, 128, 64, 32, 16]),
        (TemporalDecoder, 'src/unet5/unet_decoder.py', [256, 128, 64, 32, 32]),
    ]:
        raw = (source_root / source).read_bytes()
        pin = next(f for f in pins['files'] if f['path'] == source)
        if hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('Source decoder hash mismatch')
        ns = {}
        exec(compile(raw, '<pinned-decoder>', 'exec'), ns)
        for training in (False, True):
            for dropout in (0., .25):
                torch.manual_seed(124)
                actual = cls([64, 64, 128, 256, 512], channels, dropout=dropout)
                expected = ns['UnetDecoder']([64, 64, 128, 256, 512], channels, dropout=dropout)
                expected.load_state_dict(actual.state_dict(), strict=True)
                actual.train(training)
                expected.train(training)
                features = [torch.randn(2, c, s, s, requires_grad=True)
                            for c, s in zip([64, 64, 128, 256, 512], [32, 16, 8, 4, 2])]
                oracle_features = [x.detach().clone().requires_grad_() for x in features]
                torch.manual_seed(643)
                y = actual(features)
                torch.manual_seed(643)
                oracle = expected(oracle_features)
                torch.testing.assert_close(y, oracle, rtol=0, atol=0)
                side = 32 if cls is SingleDecoder else 64
                assert y.shape == (2, 32, side, side)
                checks['forward_cases'] += 1
                y.square().mean().backward()
                oracle.square().mean().backward()
                for a, b in zip(features, oracle_features):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                    checks['input_gradient_cases'] += 1
                for (name, a), (other, b) in zip(actual.named_parameters(), expected.named_parameters()):
                    assert name == other
                    if a.grad is None or b.grad is None:
                        assert a.grad is None and b.grad is None
                    else:
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                    checks['parameter_gradient_cases'] += 1
                for name, value in actual.state_dict().items():
                    torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
                checks['checkpoint_and_batchnorm_cases'] += 1
                if cls is SingleDecoder:
                    assert all(p.grad is None for p in actual.blocks[-1].parameters())
                    assert all(p.grad is not None for p in actual.blocks[-2].parameters())
                    checks['single_branch_skipped_block_cases'] += 1
    paths = ['sciona/contrails_single_decoder.py', 'sciona/contrails_temporal_decoder.py',
             'scripts/validate_contrails_decoders.py']
    return {'approved': False, 'source_commit': pins['commit'], 'checks': checks,
            'versions': {p: importlib.metadata.version(p) for p in
                         ['torch', 'torchvision', 'timm', 'segmentation-models-pytorch']},
            'hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
            'scope': 'Full source decoder widths; synthetic spatial pyramid. Encoder image execution and full training lifecycle not covered.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate(root, args.source_root)
    (root / 'docs/reviews/competition_contrails_decoders.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['checks']))
