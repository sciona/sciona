"""Full-resolution uninitialized APTOS model diagnostics, synthetic only."""
import gc
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import torch
import pretrainedmodels
from sciona.aptos_models import FAMILIES, build_uninitialized
from sciona.aptos_pooling import GeM


def main():
    torch.set_num_threads(2)
    torch.manual_seed(147)
    rows = {}
    for family, (factory, pool, width, size) in FAMILIES.items():
        with patch('torch.utils.model_zoo.load_url', side_effect=AssertionError('Implicit model download')):
            model = build_uninitialized(family).eval()
        assert isinstance(getattr(model, pool), GeM)
        assert model.last_linear.in_features == width and model.last_linear.out_features == 1
        assert all(parameter.requires_grad for parameter in model.parameters())
        x = torch.linspace(-1., 1., 3 * size * size).reshape(1, 3, size, size)
        with torch.no_grad():
            output = model(x)
            features = model.features(x)
            # Independent spatial reduction checks actual pool routing.
            exponent = getattr(model, pool).p
            pooled = features.clamp(min=1e-6).pow(exponent).mean((-2, -1)).pow(1. / exponent)
            expected = torch.nn.functional.linear(pooled, model.last_linear.weight, model.last_linear.bias)
        assert output.shape == (1, 1) and torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
        source = Path(inspect.getfile(getattr(pretrainedmodels, factory)))
        rows[family] = {'input_size': size, 'output_shape': list(output.shape),
            'pool_attribute': pool, 'parameters': sum(p.numel() for p in model.parameters()),
            'independent_pool_head_comparison': True, 'all_parameters_trainable': True,
            'backbone_module_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        print('PASS uninitialized full-resolution model', family, flush=True)
        del model, x, output, features, pooled, expected
        gc.collect()
    paths = ['sciona/aptos_models.py', 'sciona/aptos_pooling.py', 'scripts/validate_aptos_models.py']
    report = {'approved': False, 'catalog_mutations': 0, 'passed': True,
        'synthetic_only': True, 'pretrained_weights_used': False, 'families': rows,
        'scope': 'Actual full-resolution evaluation with random initialization; no trained, pretrained, winning-head or full-workflow parity claim.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_aptos_models_validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
