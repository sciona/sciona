"""Full-resolution ResNeXt train/validation parity with isolated winning helpers."""
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import timm
import torch
from torch import nn

from sciona.cassava_resnext import Classifier, optimizer_for, train_epoch, validate
from scripts.validate_cassava_source_components import source_helpers


def main():
    torch.set_num_threads(4)
    source = Path('/private/tmp/sciona_cassava_winner_source/resnext.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['resnext']['code_sha256']
    ns = source_helpers(source, {'CustomResNext', 'AverageMeter', 'asMinutes', 'timeSince', 'train_fn', 'valid_fn'},
        {'torch': torch, 'nn': nn, 'timm': timm, 'np': np, 'math': math, 'time': time,
         'CFG': SimpleNamespace(target_size=5, gradient_accumulation_steps=1, max_grad_norm=1000, print_freq=100)})
    torch.manual_seed(294)
    model = Classifier(pretrained=False)
    reference = ns['CustomResNext'](pretrained=False)
    reference.load_state_dict(model.state_dict())
    assert list(model.state_dict()) == list(reference.state_dict())
    batches = [(torch.randn(n, 3, 512, 512), torch.arange(n) % 5) for n in (2, 1)]
    optimizer = optimizer_for(model)
    ref_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=False)
    actual_train = train_epoch(model, batches, optimizer)
    with contextlib.redirect_stdout(io.StringIO()):
        expected_train = ns['train_fn'](batches, reference, nn.CrossEntropyLoss(), ref_optimizer, 0, None, 'cpu')
    np.testing.assert_allclose(actual_train, expected_train, atol=1e-7)
    for key, actual in model.state_dict().items():
        torch.testing.assert_close(actual, reference.state_dict()[key], atol=1e-7, rtol=1e-6)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        for key in ['step', 'exp_avg', 'exp_avg_sq']:
            torch.testing.assert_close(optimizer.state[a][key], ref_optimizer.state[b][key], atol=1e-7, rtol=1e-6)
    actual_loss, actual_probs = validate(model, batches)
    with contextlib.redirect_stdout(io.StringIO()):
        expected_loss, expected_probs = ns['valid_fn'](batches, reference, nn.CrossEntropyLoss(), 'cpu')
    np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-7)
    np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-7)
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'pretrained_weights_used': False, 'parameters': sum(p.numel() for p in model.parameters()),
              'training_batches': 2, 'validation_examples': 3,
              'source_weights_and_adam_state_match': True, 'source_validation_match': True,
              'runtime': {p: importlib.metadata.version(p) for p in ['torch', 'timm', 'numpy']},
              'scope': 'Full-resolution architecture and one-epoch diagnostic; historical timm, pretrained weights, preprocessing and full lifecycle remain unverified.'}
    paths = ['sciona/cassava_pretrained.py', 'sciona/cassava_resnext.py', 'scripts/validate_cassava_resnext.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_resnext_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
