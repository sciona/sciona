"""One real full-backbone update with pinned weights and synthetic inputs.

This establishes training mechanics, not the full eight-model/two-stage run.
Adam with constant 1e-4 is an explicit reference choice, not recovered source.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sciona.aptos_initialization import build_from_checkpoint
from sciona.aptos_models import FAMILIES
from sciona.aptos_preprocessing import prepare_rgb
from sciona.aptos_training import train_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', required=True, type=Path)
    parser.add_argument('--family', required=True, choices=FAMILIES)
    args = parser.parse_args()
    pins_path = Path('docs/reviews/competition_aptos_pretrained_mirror_pins.json')
    pins = json.loads(pins_path.read_text())['artifacts'][args.family]
    torch.set_num_threads(2)
    torch.manual_seed(731)
    model = build_from_checkpoint(args.family, args.artifact_root / (args.family + '.safetensors'),
                                 expected_sha256=pins['sha256'], checkpoint_format='safetensors')
    synthetic = (np.arange(31 * 47 * 3).reshape(31, 47, 3) % 256).astype(np.uint8)
    x = torch.stack([prepare_rgb(synthetic, args.family), prepare_rgb(255 - synthetic, args.family)])
    y = torch.tensor([[.75], [3.25]])
    pool = FAMILIES[args.family][1]
    names = [next(iter(dict(model.named_parameters()))), pool + '.p', 'last_linear.weight']
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().clone() for name in names}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = train_epochs(model, optimizer, lambda epoch: [(x, y)], epochs=1)
    changes = {name: float((parameters[name].detach() - before[name]).abs().max()) for name in names}
    assert all(value > 0 for value in changes.values())
    assert len(optimizer.state) == len(parameters)
    paths = ['sciona/aptos_initialization.py', 'sciona/aptos_models.py', 'sciona/aptos_pooling.py',
             'sciona/aptos_preprocessing.py', 'sciona/aptos_training.py',
             'scripts/validate_aptos_training.py', str(pins_path)]
    report = {'passed': True, 'approved': False, 'catalog_mutations': 0,
        'synthetic_inputs_only': True, 'family': args.family, 'artifact_sha256': pins['sha256'],
        'epochs': 1, 'optimizer_steps': 1, 'batch_size': 2,
        'full_resolution': FAMILIES[args.family][3], 'loss': 'SmoothL1Loss beta=1 mean',
        'optimizer_reference_choice': 'Adam default betas/epsilon, constant learning rate 1e-4',
        'finite_gradients_all_parameter_tensors': len(parameters),
        'changed_probe_max_absolute_deltas': changes, 'aggregate_losses': losses,
        'scope': 'Actual pretrained backbone, GeM exponent and scalar head update. No augmentation, complete stage budget or eight-model lifecycle qualification.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_aptos_training_' + args.family + '_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS actual pretrained full-resolution training update', args.family, flush=True)


if __name__ == '__main__':
    main()
