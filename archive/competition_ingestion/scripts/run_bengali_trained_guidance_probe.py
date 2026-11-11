"""Winner-budget synthetic CycleGAN fit using a checked synthetic B0 checkpoint.

One synthetic branch only. This does not stand in for either final font branch
or for the full seen/OOD/routing pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet

from scripts.qualify_bengali_classifier_checkpoint import validate_completion
from sciona.bengali_checkpoint_handoff import load_frozen_classifier
from sciona.bengali_fit import fit_population
from sciona.bengali_font_classifier import BengaliFontClassifier
from sciona.bengali_font_sampling import FontParameterSampler
from sciona.bengali_gan_networks import BengaliGenerator, BengaliDiscriminator, initialize_image_network
from sciona.bengali_gan_training import CycleGANTraining
from sciona.bengali_population import ImagePopulation
from sciona.bengali_preprocessing import prepare_handwriting
from sciona.bengali_sampling import SamplingBudget


def validate_handoff(receipt_path, expected_sha256, classifier_runtime, completion_path, root):
    payload = receipt_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('classifier consistency receipt content changed')
    receipt = json.loads(payload)
    if (receipt.get('receipt_kind') != 'post_completion_synthetic_consistency'
            or receipt.get('passed') is not True or receipt.get('synthetic_only') is not True
            or receipt.get('approved') is not False or receipt.get('catalog_mutations') != 0
            or receipt.get('selected_epoch_evaluation_replayed') is not True
            or receipt.get('training_time_checkpoint_digest_available') is not False):
        raise ValueError('completed synthetic consistency receipt required')
    completion = completion_path.read_bytes()
    history = (classifier_runtime / 'history.json').read_bytes()
    if (hashlib.sha256(completion).hexdigest() != receipt.get('completion_sha256')
            or hashlib.sha256(history).hexdigest() != receipt.get('history_sha256')):
        raise ValueError('classifier completion evidence changed after qualification')
    report = json.loads(completion)
    epoch, accuracy = validate_completion(report, json.loads(history), root)
    if (receipt.get('classifier_implementation_sha256') != report['implementation_sha256']
            or receipt.get('selected_epoch') != epoch or receipt.get('same_population_accuracy') != accuracy):
        raise ValueError('classifier receipt disagrees with training evidence')
    verifiers = receipt.get('verifier_sha256', {})
    expected = {'scripts/qualify_bengali_classifier_checkpoint.py', 'sciona/bengali_checkpoint_handoff.py'}
    if set(verifiers) != expected:
        raise ValueError('classifier verifier binding inventory differs')
    for name, digest in verifiers.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('classifier verifier changed after qualification')
    checkpoint = classifier_runtime / 'best.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != receipt.get('checkpoint_sha256'):
        raise ValueError('classifier checkpoint changed after qualification')
    return receipt


def main(runtime, output, classifier_runtime, completion, receipt_path, receipt_sha256):
    root = Path(__file__).resolve().parents[1]
    receipt = validate_handoff(receipt_path, receipt_sha256, classifier_runtime, completion, root)
    if runtime.exists() or output.exists():
        raise ValueError('fresh CycleGAN runtime and output required')
    files = sorted((root / 'sciona').glob('bengali_*.py')) + [Path(__file__).resolve(),
        root / 'scripts/qualify_bengali_classifier_checkpoint.py']
    bindings = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    torch.set_num_threads(2)
    torch.manual_seed(1983)
    classifier = BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0'))
    load_frozen_classifier(classifier, classifier_runtime / 'best.pt',
                           expected_sha256=receipt['checkpoint_sha256'])
    frozen = {k: v.clone() for k, v in classifier.state_dict().items()}
    # Independently sampled synthetic domains with aligned intensity classes.
    labels = np.arange(64, dtype=np.int64) % 8
    rng = np.random.default_rng(953)
    populations = []
    for domain in range(2):
        images = np.empty((64, 137, 236), dtype=np.uint8)
        for index, label in enumerate(labels):
            images[index] = np.clip(24 + int(label) * 27 + domain * 3
                + rng.integers(0, 12, (137, 236)), 0, 255).astype(np.uint8)
        populations.append(ImagePopulation(images, labels))
    hand, font = populations
    networks = [BengaliGenerator(), BengaliGenerator(), BengaliDiscriminator(), BengaliDiscriminator()]
    for index, network in enumerate(networks):
        initialize_image_network(network, generator=torch.Generator().manual_seed(933 + index))
    before = [next(n.parameters()).detach().clone() for n in networks]
    budget = SamplingBudget(64, 64, 32, 40)
    trainer = CycleGANTraining(generator_a=networks[0], generator_b=networks[1],
        discriminator_a=networks[2], discriminator_b=networks[3], classifier=classifier,
        classifier_weight=4., total_steps=budget.total_steps, replay_seed_a=941, replay_seed_b=942)
    print(json.dumps(dict(started=True, synthetic_only=True, epochs=40, batch_size=32,
                          optimizer_updates_per_optimizer=80, classifier_weight=4.)), flush=True)
    result = fit_population(trainer, hand, font, budget, hand_seed=943, font_seed=944,
        font_sampler=FontParameterSampler(python_seed=945, image_seed=946, backend='sfc64'),
        output_directory=runtime)
    if result['completed_epochs'] != 40 or result['completed_steps'] != 80:
        raise ValueError('incomplete CycleGAN budget')
    if not all(not torch.equal(old, next(n.parameters())) for old, n in zip(before, networks)):
        raise ValueError('a trainable network did not update')
    if not all(torch.isfinite(p).all() for n in networks for p in n.parameters()):
        raise ValueError('nonfinite trained network')
    for name, value in classifier.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    for entry in result['history']:
        if hashlib.sha256((runtime / entry['generator_file']).read_bytes()).hexdigest() != entry['generator_sha256']:
            raise ValueError('saved epoch generator changed')
    final = result['history'][-1]
    fresh = BengaliGenerator().eval()
    fresh.load_state_dict(torch.load(runtime / final['generator_file'], weights_only=True))
    with torch.no_grad():
        probe = prepare_handwriting(hand.images[:1])
        torch.testing.assert_close(fresh(probe), trainer.generator_b(probe), rtol=0, atol=0)
    if bindings != {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}:
        raise ValueError('CycleGAN implementation changed during training')
    validate_handoff(receipt_path, receipt_sha256, classifier_runtime, completion, root)
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        completed_epochs=40, batch_size=32, optimizer_updates_per_optimizer=80, classifier_weight=4.,
        synthetic_classifier_receipt_sha256=receipt_sha256, implementation_sha256=bindings,
        all_four_trainable_networks_updated=True, frozen_classifier_unchanged=True,
        fresh_generator_output_exact=True, verified_epoch_checkpoints=40,
        limits=['One synthetic intensity branch; neither final font branch nor glyph-recognition validation.',
                'Source epoch/batch controls with a smaller synthetic population; not the original optimizer update count.',
                'Classifier handoff uses post-completion consistency evidence, not a training-time digest.',
                'Full seen/OOD/routing pipeline, historical environment, licensing and publication gates remain pending.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'implementation_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classifier-runtime', type=Path, required=True)
    parser.add_argument('--completion-report', type=Path, required=True)
    parser.add_argument('--classifier-receipt', type=Path, required=True)
    parser.add_argument('--classifier-receipt-sha256', required=True)
    args = parser.parse_args()
    main(args.runtime_directory, args.output, args.classifier_runtime, args.completion_report,
         args.classifier_receipt, args.classifier_receipt_sha256)
