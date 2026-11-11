"""Exercise full B7 classifier updates on synthetic tensors, without publication."""
import gc
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path

from efficientnet_pytorch import EfficientNet
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from sciona.grapheme_b7_head import GraphemeB7Head
from sciona.grapheme_b7_training import make_optimizer, train_batch


FILES = [
    'scripts/validate_bengali_b7_training_step.py',
    'sciona/grapheme_b7_head.py', 'sciona/grapheme_b7_training.py',
    'sciona/grapheme_b7_schedule.py', 'sciona/bengali_ood_loss.py',
]


def hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = hashes()
    results = []
    with threadpool_limits(limits=1):
        for branch, classes in [('ood', 1295), ('seen', 14784)]:
            torch.manual_seed(1701)
            model = GraphemeB7Head(EfficientNet.from_name('efficientnet-b7'), branch=branch)
            optimizer = make_optimizer(model)
            images = torch.randn(1, 3, 137, 236)
            labels = torch.tensor([classes - 1], dtype=torch.int64)
            tracked = dict(stem=model.backbone._conv_stem.weight, head=model.fc.weight)
            initial = {name: value.detach().clone() for name, value in tracked.items()}
            bn = model.backbone._bn0.num_batches_tracked.item()
            updates = []
            for index in (1, 2):
                result = train_batch(model, optimizer, images, labels, branch=branch,
                                     schedule_index=index, total_steps=5, warmup_steps=2)
                absent = [name for name, p in model.named_parameters() if p.grad is None]
                assert absent == ['backbone._fc.weight', 'backbone._fc.bias'], absent
                assert all(torch.isfinite(p).all() for p in model.parameters())
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                assert all(value.grad.abs().sum().item() > 0 for value in tracked.values())
                assert len(optimizer.state) == len(list(model.parameters())) - 2
                assert all(state['step'].item() == index for state in optimizer.state.values())
                updates.append(result)
            deltas = {name: (value.detach() - initial[name]).abs().max().item()
                      for name, value in tracked.items()}
            assert all(value > 0 for value in deltas.values())
            assert model.backbone._bn0.num_batches_tracked.item() - bn == 2
            result = dict(branch=branch, updates=updates, maximum_parameter_changes=deltas,
                          trainable_parameters=sum(p.numel() for p in model.parameters()),
                          optimizer_parameter_tensors=len(optimizer.state),
                          unused_parameter_tensors=absent, batch_norm_updates=2,
                          finite_parameters_and_gradients=True, backbone_frozen=False)
            results.append(result)
            print(json.dumps(result), flush=True)
            del model, optimizer, images, labels, tracked, initial
            gc.collect()
        pools = threadpool_info()
        assert all(pool['num_threads'] == 1 for pool in pools)
    assert hashes() == before, 'Software changed during validation'
    report = dict(format='bengali-b7-training-step.v1', passed=True, approved=False,
                  catalog_mutations=0, synthetic_only=True, branches=results,
                  implementation_sha256=before,
                  installed_backbone_module_sha256=hashlib.sha256(
                      Path(inspect.getfile(EfficientNet)).read_bytes()).hexdigest(),
                  versions={name: version(name) for name in ['torch', 'efficientnet-pytorch']},
                  resources=dict(torch_threads=1, torch_interop_threads=1,
                                 native_pool_threads=[p['num_threads'] for p in pools],
                                 data_loader_workers=0, branches_serial=True),
                  limits=[
                      'Two updates per branch with random initialization and synthetic input; not full training or accuracy evidence.',
                      'Caller-selected schedule indices and input distribution do not establish historical warmup, scheduler ordering or normalization.',
                      'Installed B7 implementation is exercised; historical dependency and pretrained checkpoint fidelity are not established.',
                      'CycleGAN integration, threshold calibration and complete workflow qualification remain separate requirements.',
                  ])
    Path('docs/reviews/bengali_b7_training_step.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
