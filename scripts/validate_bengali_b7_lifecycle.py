"""Run both 200-epoch B7 lifecycles on synthetic inputs; no catalog approval."""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from sciona.bengali_ood_loss import ood_training_loss
from sciona.grapheme_b7_fit import fit_classifier
from sciona.grapheme_b7_head import GraphemeB7Head
from sciona.grapheme_b7_initialization import load_standard_b7, STANDARD_B7_SHA256
from sciona.grapheme_b7_preprocessing import prepare_classifier_images
from sciona.svhn_autoaugment import SVHNAutoAugment


FILES = ['scripts/validate_bengali_b7_lifecycle.py', 'sciona/grapheme_b7_fit.py',
         'sciona/grapheme_b7_training.py', 'sciona/grapheme_b7_head.py',
         'sciona/grapheme_b7_initialization.py', 'sciona/grapheme_b7_preprocessing.py',
         'sciona/grapheme_b7_schedule.py', 'sciona/bengali_ood_loss.py',
         'sciona/svhn_autoaugment.py']


def hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES}


def fixture(seed):
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 32, (2, 137, 236), dtype=np.uint8)
    images[0, 30:100, 60:90] += 192
    images[1, 50:80, 40:180] += 192
    return images


def main(checkpoint, runtime, output):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = hashes()
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != STANDARD_B7_SHA256:
        raise ValueError('Publisher initialization content differs')
    runtime.mkdir(parents=True, exist_ok=False)
    results = []
    with threadpool_limits(limits=1):
        for branch in ('ood', 'seen'):
            torch.manual_seed(1703)
            model = GraphemeB7Head(load_standard_b7(checkpoint), branch=branch)
            initial_stem = model.backbone._conv_stem.weight.detach().clone()
            initial_head = model.fc.weight.detach().clone()
            initial_bn = model.backbone._bn0.num_batches_tracked.item()
            augmentation = SVHNAutoAugment(seed=1704)
            train_images, validation_images = fixture(1705), fixture(1706)
            labels = torch.tensor([0, 1], dtype=torch.int64)
            normalization = dict(mean=(.5, .5, .5), std=(.5, .5, .5))
            validation = prepare_classifier_images(validation_images, augmentation=None, **normalization)
            def batches(epoch):
                yield prepare_classifier_images(train_images, augmentation=augmentation, **normalization), labels
            def validate(candidate, epoch):
                assert not candidate.training and not torch.is_grad_enabled()
                logits = candidate(validation)
                loss = (ood_training_loss(logits, labels) if branch == 'ood'
                        else torch.nn.functional.cross_entropy(logits, labels))
                if (epoch + 1) % 10 == 0:
                    print(json.dumps(dict(branch=branch, completed_epochs=epoch+1,
                                          validation_loss=float(loss))), flush=True)
                return -float(loss)
            result = fit_classifier(model, branch=branch, epochs=200, steps_per_epoch=1,
                                    batches_for_epoch=batches, validate=validate, warmup_steps=10,
                                    first_schedule_index=0, checkpoint_policy='last',
                                    output_directory=runtime/branch)
            assert result['completed_epochs'] == result['completed_steps'] == 200
            assert result['selected_checkpoint']['epoch'] == 199
            assert all(p.requires_grad for p in model.parameters())
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert model.backbone._bn0.num_batches_tracked.item() - initial_bn == 200
            changes = dict(stem=float((model.backbone._conv_stem.weight.detach()-initial_stem).abs().max()),
                           head=float((model.fc.weight.detach()-initial_head).abs().max()))
            assert min(changes.values()) > 0
            model.eval()
            with torch.no_grad():
                expected = model(validation)
            del model, initial_stem, initial_head
            gc.collect()
            selected = runtime/branch/'selected.pt'
            assert hashlib.sha256(selected.read_bytes()).hexdigest() == result['selected_checkpoint']['sha256']
            restored = GraphemeB7Head(EfficientNet.from_name('efficientnet-b7'), branch=branch)
            restored.load_state_dict(torch.load(selected, map_location='cpu', weights_only=True), strict=True)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(validation), expected, rtol=0, atol=0)
            results.append(dict(branch=branch, completed_epochs=200, optimizer_updates=200,
                                batch_norm_updates=200, selected_epoch=199,
                                exact_checkpoint_replay=True, maximum_parameter_changes=changes,
                                initial_training_loss=result['history'][0]['training_loss'],
                                final_training_loss=result['history'][-1]['training_loss'],
                                final_validation_score=result['history'][-1]['validation_score'],
                                all_epoch_metrics_finite=all(np.isfinite(h['training_loss']) and
                                    np.isfinite(h['validation_score']) for h in result['history'])))
            assert results[-1]['all_epoch_metrics_finite']
            del restored, expected, result
            gc.collect()
        pools = threadpool_info()
        assert all(p['num_threads'] == 1 for p in pools)
    assert hashes() == before, 'Bound software changed during execution'
    report = dict(format='bengali-b7-lifecycle.v1', passed=True, approved=False, catalog_mutations=0,
                  synthetic_only=True, branches=results, implementation_sha256=before,
                  initialization_sha256=STANDARD_B7_SHA256,
                  versions={name: version(name) for name in ['torch', 'efficientnet-pytorch', 'numpy', 'pillow']},
                  policy=dict(epochs=200, batch_size=2, steps_per_epoch=1, warmup_steps=10,
                              first_schedule_index=0, checkpoint='last', mean=[.5]*3, std=[.5]*3),
                  resources=dict(compute_threads=1, torch_interop_threads=1, data_loader_workers=0,
                                 branches_serial=True, native_pool_threads=[p['num_threads'] for p in pools]),
                  limits=[
                      'Synthetic two-example populations and batch 2 instead of source batch 32; no recognition or transfer-accuracy claim.',
                      '200 epochs match the published count; normalization, warmup, schedule origin, validation metric and final checkpoint policy are explicit synthetic-run choices.',
                      'Publisher initialization candidate and installed libraries do not establish exact historical artifact/environment identity.',
                      'Both font/generator branches, routing calibration and complete stored graph qualification remain pending.'
                  ])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False, catalog_mutations=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.runtime, args.output)
