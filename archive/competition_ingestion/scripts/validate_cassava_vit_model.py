"""Actual full ViT epoch and validation against isolated winning helpers."""
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import timm
import torch
from torch import nn

from sciona.cassava_vit_model import Classifier, validate
from sciona.cassava_vit_epoch import train_epoch
from sciona.cassava_vit_ordered_loss import loss as ordered_loss
from scripts.validate_cassava_source_components import source_helpers
from scripts.validate_cassava_vit_epoch import Progress


def main():
    torch.set_num_threads(4)
    source = Path('/private/tmp/sciona_cassava_winner_source/vit.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['vit']['code_sha256']
    ns = source_helpers(source, {'CassvaImgClassifier', 'log_t', 'exp_t', 'compute_normalization_fixed_point',
        'compute_normalization_binary_search', 'ComputeNormalization', 'compute_normalization',
        'tempered_softmax', 'bi_tempered_logistic_loss', 'train_one_epoch', 'valid_one_epoch'},
        {'torch': torch, 'nn': nn, 'timm': timm, 'np': np, 'time': time, 'gc': gc, 'tqdm': Progress,
         'autocast': contextlib.nullcontext,
         'CFG': {'t1': .8, 't2': 1.4, 'smoothing': .06, 'accum_iter': 2},
         'xm': SimpleNamespace(optimizer_step=lambda opt: opt.step(), mark_step=lambda: None,
                               mesh_reduce=lambda name, value, reduction: value, master_print=lambda *a: None)})
    torch.manual_seed(832)
    model = Classifier(pretrained=False)
    reference = ns['CassvaImgClassifier']('vit_base_patch16_384', 5, pretrained=False)
    reference.load_state_dict(model.state_dict())
    batches = [(torch.randn(1, 3, 384, 384), torch.tensor([label])) for label in (1, 3)]
    opt = torch.optim.Adam(model.parameters(), lr=1e-4 / 7)
    ref_opt = torch.optim.Adam(reference.parameters(), lr=1e-4 / 7)
    result = train_epoch(model, batches, opt, loss_function=ordered_loss)
    ns['train_one_epoch'](0, reference, ref_opt, batches, 'cpu')
    assert result['optimizer_steps'] == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=2e-6, rtol=2e-5)
    actual_loss, actual_acc = validate(model, batches)
    expected_loss, expected_acc = ns['valid_one_epoch'](0, reference, batches, 'cpu')
    np.testing.assert_allclose(actual_loss, expected_loss.detach().numpy(), atol=2e-6, rtol=2e-5)
    assert actual_acc == expected_acc
    last_only, _ = validate(model, batches[-1:])
    assert actual_loss == last_only
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'pretrained_weights_used': False, 'parameters': sum(p.numel() for p in model.parameters()),
              'full_resolution_microbatches': 2, 'accumulated_optimizer_steps': 1,
              'source_weight_match': True, 'source_validation_match': True,
              'last_batch_validation_loss_preserved': True,
              'runtime': {p: importlib.metadata.version(p) for p in ['torch', 'timm', 'numpy']},
              'scope': 'Full ViT CPU epoch diagnostic; no historical TPU reduction, pretrained provenance or full fold qualification.'}
    paths = ['sciona/cassava_pretrained.py', 'sciona/cassava_vit_model.py', 'sciona/cassava_vit_ordered_loss.py', 'sciona/cassava_vit_epoch.py', 'sciona/cassava_loss.py',
             'scripts/validate_cassava_vit_model.py', 'scripts/validate_cassava_vit_epoch.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_vit_model_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
