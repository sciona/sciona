"""Audit saved winner-budget synthetic CycleGAN epochs without certifying the fit."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from sciona.bengali_gan_networks import BengaliGenerator


METRICS = {'generator/' + key for key in ('gan_a', 'gan_b', 'cycle_a', 'cycle_b', 'font', 'total')}
METRICS |= {'discriminator/' + key for key in ('real_a', 'fake_a', 'real_b', 'fake_b', 'domain_a', 'domain_b', 'total')}


def audit_history(history, runtime, expected_state):
    if not isinstance(history, list) or not 1 <= len(history) <= 40:
        raise ValueError('one through40recorded epochs required')
    for index, row in enumerate(history):
        if (type(row.get('epoch')) is not int or row['epoch'] != index
                or type(row.get('steps')) is not int or row['steps'] != 2):
            raise ValueError('synthetic source epoch/update budget differs')
        metrics = row.get('metrics', {})
        if set(metrics) != METRICS or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in metrics.values()):
            raise ValueError('invalid recorded CycleGAN metrics')
        identities = {
            'generator/total': sum(metrics['generator/' + k] for k in ('gan_a', 'gan_b', 'cycle_a', 'cycle_b', 'font')),
            'discriminator/domain_a': (metrics['discriminator/real_a'] + metrics['discriminator/fake_a']) / 2,
            'discriminator/domain_b': (metrics['discriminator/real_b'] + metrics['discriminator/fake_b']) / 2,
            'discriminator/total': metrics['discriminator/domain_a'] + metrics['discriminator/domain_b'],
        }
        if any(not math.isclose(metrics[k], v, rel_tol=1e-5, abs_tol=1e-5) for k, v in identities.items()):
            raise ValueError('recorded loss decomposition differs')
        name = f'generator_epoch_{index + 1}.pt'
        if row.get('generator_file') != name:
            raise ValueError('epoch checkpoint reference differs')
        payload = (runtime / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row.get('generator_sha256'):
            raise ValueError('epoch checkpoint content differs from history')
        state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or set(state) != set(expected_state):
            raise ValueError('epoch generator tensor inventory differs')
        for key, value in state.items():
            target = expected_state[key]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != target.shape or value.dtype != target.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError('epoch generator tensor contract differs')
    return dict(completed_epochs=len(history), recorded_generator_updates=len(history) * 2,
                recorded_discriminator_updates=len(history) * 2,
                verified_epoch_checkpoints=len(history), recorded_epoch_budget_complete=len(history) == 40)


def main(runtime, output):
    torch.set_num_threads(2)
    # Atomic history writes make this a stable prefix even as more epochs finish.
    raw = (runtime / 'history.json').read_bytes()
    result = audit_history(json.loads(raw), runtime, BengaliGenerator().state_dict())
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        **result, history_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Recorded history, loss decomposition and generator checkpoint consistency only; no independent training replay.',
                'Discriminators, optimizers and replay pools are not present in source epoch inference checkpoints.',
                'Final fit report must establish unchanged code/classifier evidence, all-network updates and output replay.',
                'One synthetic branch; complete Bengali pipeline and publication qualification remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.runtime_directory, args.output)
