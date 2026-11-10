"""Connect two synthetic B0-to-CycleGAN branches and their marginal ensemble.

Constructed geometric domains are not historical fonts or recognition evidence.
The original full graph and routing calibration remain separate requirements.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import cv2
from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from sciona.bengali_checkpoint_handoff import load_frozen_classifier
from sciona.bengali_classifier_fit import fit_font_classifier
from sciona.bengali_fit import fit_population
from sciona.bengali_font_classifier import BengaliFontClassifier
from sciona.bengali_font_sampling import FontParameterSampler
from sciona.bengali_gan_networks import BengaliGenerator, BengaliDiscriminator, initialize_image_network
from sciona.bengali_gan_training import CycleGANTraining
from sciona.bengali_population import ImagePopulation
from sciona.bengali_preprocessing import prepare_handwriting
from sciona.bengali_pretraining_sampling import PretrainingParameterSampler
from sciona.bengali_sampling import SamplingBudget
from sciona.bengali_unseen_ensemble import marginal_predictions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def populations():
    labels = np.arange(4, dtype=np.int64) % 2
    domains = []
    for domain in range(3):
        rng = np.random.default_rng(2400 + domain)
        images = rng.integers(0, 24, (4, 137, 236), dtype=np.uint8)
        for index, label in enumerate(labels):
            if label == 0:
                images[index, 25:112, 85:145] += 200
                if domain == 2:
                    images[index, 35:102, 95:135] -= 180
            else:
                images[index, 45:95, 35:200] += 200
                if domain == 2:
                    images[index, 55:85, 45:190] -= 180
        if domain == 1:
            images = np.roll(images, 5, axis=2)
        domains.append(ImagePopulation(images, labels))
    return domains


def main(runtime, output):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    assert cv2.getNumThreads() == 1
    files = sorted(Path('sciona').glob('bengali_*.py')) + [Path(__file__).relative_to(Path.cwd())]
    bindings = {str(path): sha(path) for path in files}
    runtime.mkdir(parents=True, exist_ok=False)
    hand, *fonts = populations()
    probe = prepare_handwriting(hand.images[:2])
    predictions, replays, results = [], [], []
    with threadpool_limits(limits=1):
        for index, font in enumerate(fonts):
            torch.manual_seed(2500 + index)
            branch = runtime/str(index)
            branch.mkdir()
            classifier = BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0'))
            print(json.dumps(dict(branch=index, stage='classifier_started')), flush=True)
            fitted = fit_font_classifier(classifier, font, epochs=60, batch_size=2,
                train_seed=2510+index, validation_seed=2520+index,
                sampler=PretrainingParameterSampler(python_seed=2530+index, image_seed=2540+index, backend='sfc64'),
                output_directory=branch/'classifier')
            assert fitted['completed_epochs'] == 60 and fitted['optimizer_updates'] == 120
            selected = fitted['best_checkpoint']
            if selected is None:
                raise ValueError('Source best-score policy selected no classifier checkpoint')
            checkpoint = branch/'classifier'/selected['file']
            classifier = load_frozen_classifier(classifier, checkpoint, expected_sha256=selected['sha256'])
            frozen = {key: value.clone() for key, value in classifier.state_dict().items()}
            networks = [BengaliGenerator(), BengaliGenerator(), BengaliDiscriminator(), BengaliDiscriminator()]
            for offset, network in enumerate(networks):
                initialize_image_network(network, generator=torch.Generator().manual_seed(2600+index*10+offset))
            initial = [next(network.parameters()).detach().clone() for network in networks]
            budget = SamplingBudget(4, 4, 2, 40)
            trainer = CycleGANTraining(generator_a=networks[0], generator_b=networks[1],
                discriminator_a=networks[2], discriminator_b=networks[3], classifier=classifier,
                classifier_weight=4., total_steps=budget.total_steps,
                replay_seed_a=2700+index, replay_seed_b=2710+index)
            print(json.dumps(dict(branch=index, stage='classifier_completed_gan_started',
                                  classifier_epochs=60, classifier_updates=120)), flush=True)
            fitted_gan = fit_population(trainer, hand, font, budget,
                hand_seed=2720+index, font_seed=2730+index,
                font_sampler=FontParameterSampler(python_seed=2740+index, image_seed=2750+index, backend='sfc64'),
                output_directory=branch/'gan')
            assert fitted_gan['completed_epochs'] == 40 and fitted_gan['completed_steps'] == 80
            assert all(not torch.equal(old, next(network.parameters())) for old, network in zip(initial, networks))
            assert all(torch.isfinite(p).all() for network in networks for p in network.parameters())
            for key, value in classifier.state_dict().items():
                torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
            for entry in fitted_gan['history']:
                assert sha(branch/'gan'/entry['generator_file']) == entry['generator_sha256']
            with torch.no_grad():
                generated = trainer.generator_b(probe)
                predicted = classifier(generated)
            predictions.append(predicted)
            final = fitted_gan['history'][-1]
            fresh_generator = BengaliGenerator().eval()
            fresh_generator.load_state_dict(torch.load(branch/'gan'/final['generator_file'], weights_only=True))
            fresh_classifier = load_frozen_classifier(
                BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0')), checkpoint,
                expected_sha256=selected['sha256'])
            with torch.no_grad():
                actual = fresh_generator(probe)
                torch.testing.assert_close(actual, generated, rtol=0, atol=0)
                replay = fresh_classifier(actual)
                torch.testing.assert_close(replay, predicted, rtol=0, atol=0)
            replays.append(replay)
            results.append(dict(branch=index, classifier_epochs=60, classifier_updates=120,
                gan_epochs=40, generator_updates=80, discriminator_updates=80,
                classifier_handoff_hash_checked=True, frozen_classifier_unchanged=True,
                all_four_gan_networks_updated=True, verified_generator_checkpoints=40,
                generated_images_and_classifier_logits_replayed_exactly=True))
            # Private synthetic handoff receipt; contains no source dataset provenance.
            (branch/'receipt.json').write_text(json.dumps(dict(classifier=selected, generator=final,
                implementation_sha256=bindings, synthetic_only=True, approved=False), indent=2)+'\n')
            print(json.dumps(results[-1]), flush=True)
            del classifier, fresh_classifier, fresh_generator, networks, trainer, frozen, initial, generated, actual
            gc.collect()
        expected = marginal_predictions(*predictions)
        actual = marginal_predictions(*replays)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        pools = threadpool_info()
        assert all(pool['num_threads'] == 1 for pool in pools)
    assert bindings == {str(path): sha(path) for path in files}, 'Bound software changed during execution'
    report = dict(format='bengali-paired-branches.v1', passed=True, approved=False, catalog_mutations=0,
        synthetic_only=True, branches=results, ensemble_replay_exact=True, prediction_count=2,
        implementation_sha256=bindings,
        versions={name: version(name) for name in ['torch', 'efficientnet-pytorch', 'numpy', 'opencv-python-headless']},
        resources=dict(compute_threads=1, interop_threads=1, opencv_threads=1, data_loader_workers=0,
                       branches_serial=True, native_pool_threads=[pool['num_threads'] for pool in pools]),
        limits=['Two constructed geometric domains, not historical fonts or glyph-recognition validation.',
                'Source epoch counts with batch 2 instead of 32 and four-example populations; original update counts are not reproduced.',
                'Random B0 initialization and installed libraries; historical artifact/environment identity remains unqualified.',
                'B7 branch integration, calibrated routing, reusable-contract review and complete stored graph publication remain pending.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False, catalog_mutations=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.runtime, args.output)
