"""Training and inference boundaries for a complete reconstructed grapheme graph.

Policies, populations and private output directories are caller-owned. B7 uses
explicit epoch shuffling and supplied normalization/schedule/checkpoint policy;
these choices do not attest to unrecovered historical settings. Returned states
bind inference checkpoints to training-time digests, never to implicit caches.
"""
import hashlib
from pathlib import Path

from efficientnet_pytorch import EfficientNet
import numpy as np
import torch

from sciona.bengali_checkpoint_handoff import load_frozen_classifier
from sciona.bengali_classifier_fit import fit_font_classifier
from sciona.bengali_fit import fit_population
from sciona.bengali_font_classifier import BengaliFontClassifier
from sciona.bengali_font_sampling import FontParameterSampler
from sciona.bengali_gan_networks import BengaliGenerator, BengaliDiscriminator, initialize_image_network
from sciona.bengali_gan_training import CycleGANTraining
from sciona.bengali_ood_loss import ood_training_loss
from sciona.bengali_population import ImagePopulation
from sciona.bengali_preprocessing import prepare_handwriting
from sciona.bengali_pretraining_sampling import PretrainingParameterSampler
from sciona.bengali_sampling import SamplingBudget
from sciona.grapheme_b7_fit import fit_classifier
from sciona.grapheme_b7_head import GraphemeB7Head
from sciona.grapheme_b7_initialization import load_standard_b7
from sciona.grapheme_b7_preprocessing import prepare_classifier_images
from sciona.svhn_autoaugment import SVHNAutoAugment


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _options(options, required):
    if not isinstance(options, dict) or set(options) != set(required):
        raise ValueError('Exact explicit training policy fields required')
    for name in ('epochs', 'batch_size'):
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError('Positive integer epoch and batch sizes required')
    if type(options['seed']) is not int or not 0 <= options['seed'] < 2**31-100:
        raise ValueError('Bounded explicit training seed required')


def _targets(values, count, classes):
    values = np.asarray(values)
    if (values.shape != (count,) or values.dtype.kind not in 'iu'
            or (values < 0).any() or (values >= classes).any()):
        raise ValueError('Aligned in-range integer classifier targets required')
    return torch.from_numpy(values.astype(np.int64, copy=True))


def train_b7(population, targets, validation, validation_targets, *, branch, options,
             initialization_path, output_directory):
    _options(options, ('epochs', 'batch_size', 'seed', 'mean', 'std', 'warmup_steps',
                       'first_schedule_index', 'checkpoint_policy'))
    if branch not in ('ood', 'seen') or not isinstance(population, ImagePopulation) or not isinstance(validation, ImagePopulation):
        raise ValueError('Explicit B7 branch and validated populations required')
    batch_size = options['batch_size']
    if len(population.images) < batch_size or len(population.images) % batch_size:
        raise ValueError('Training population must partition into complete batches')
    classes = 1295 if branch == 'ood' else 14784
    targets = _targets(targets, len(population.images), classes)
    validation_targets = _targets(validation_targets, len(validation.images), classes)
    normalization = dict(mean=options['mean'], std=options['std'])
    # Validate both input contracts before reading initialization or creating files.
    prepare_classifier_images(population.images, augmentation=None, **normalization)
    validation_input = prepare_classifier_images(validation.images, augmentation=None, **normalization)
    torch.manual_seed(options['seed'])
    model = GraphemeB7Head(load_standard_b7(initialization_path), branch=branch)
    initial = model.backbone._conv_stem.weight.detach().clone()
    policy = SVHNAutoAugment(seed=options['seed']+1)
    rng = np.random.default_rng(options['seed']+2)
    def batches(epoch):
        order = rng.permutation(len(population.images))
        for start in range(0, len(order), batch_size):
            indices = order[start:start+batch_size]
            yield prepare_classifier_images(population.images[indices], augmentation=policy,
                                            **normalization), targets[indices]
    def validate(candidate, epoch):
        logits = candidate(validation_input)
        loss = (ood_training_loss(logits, validation_targets) if branch == 'ood'
                else torch.nn.functional.cross_entropy(logits, validation_targets))
        return -float(loss)
    result = fit_classifier(model, branch=branch, epochs=options['epochs'],
        steps_per_epoch=len(population.images)//batch_size, batches_for_epoch=batches,
        validate=validate, warmup_steps=options['warmup_steps'],
        first_schedule_index=options['first_schedule_index'], checkpoint_policy=options['checkpoint_policy'],
        output_directory=output_directory)
    if torch.equal(initial, model.backbone._conv_stem.weight) or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Finite updated B7 backbone required')
    selected = result['selected_checkpoint']
    path = Path(output_directory)/'selected.pt'
    if _sha(path) != selected['sha256']:
        raise ValueError('Selected B7 checkpoint changed after training')
    return dict(kind='b7', branch=branch, path=str(path), sha256=selected['sha256'],
        normalization=normalization, metrics=dict(epochs=result['completed_epochs'],
        updates=result['completed_steps'], selected_epoch=selected['epoch'], backbone_updated=True),
        policy=dict(options))


def train_font(population, *, options, output_directory):
    _options(options, ('epochs', 'batch_size', 'seed', 'backend'))
    if not isinstance(population, ImagePopulation):
        raise ValueError('Validated font population required')
    seed = options['seed']
    sampler = PretrainingParameterSampler(python_seed=seed+1, image_seed=seed+2, backend=options['backend'])
    torch.manual_seed(seed)
    model = BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0'))
    result = fit_font_classifier(model, population, epochs=options['epochs'], batch_size=options['batch_size'],
        train_seed=seed+3, validation_seed=seed+4, sampler=sampler, output_directory=output_directory)
    selected = result['best_checkpoint']
    if selected is None:
        raise ValueError('No source-policy best classifier checkpoint selected')
    path = Path(output_directory)/selected['file']
    if _sha(path) != selected['sha256']:
        raise ValueError('Selected font checkpoint changed after training')
    return dict(kind='font', path=str(path), sha256=selected['sha256'],
                metrics=dict(epochs=result['completed_epochs'], updates=result['optimizer_updates']),
                policy=dict(options))


def train_generator(hand, font, classifier_state, *, options, output_directory):
    _options(options, ('epochs', 'batch_size', 'seed', 'backend', 'classifier_weight'))
    if not isinstance(hand, ImagePopulation) or not isinstance(font, ImagePopulation) or classifier_state.get('kind') != 'font':
        raise ValueError('Validated populations and trained font checkpoint required')
    seed = options['seed']
    sampler = FontParameterSampler(python_seed=seed+1, image_seed=seed+2, backend=options['backend'])
    budget = SamplingBudget(len(hand.images), len(font.images), options['batch_size'], options['epochs'])
    classifier = load_frozen_classifier(BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0')),
        classifier_state['path'], expected_sha256=classifier_state['sha256'])
    frozen = {key: value.clone() for key, value in classifier.state_dict().items()}
    networks = [BengaliGenerator(), BengaliGenerator(), BengaliDiscriminator(), BengaliDiscriminator()]
    for offset, network in enumerate(networks):
        initialize_image_network(network, generator=torch.Generator().manual_seed(seed+10+offset))
    initial = [next(network.parameters()).detach().clone() for network in networks]
    trainer = CycleGANTraining(generator_a=networks[0], generator_b=networks[1],
        discriminator_a=networks[2], discriminator_b=networks[3], classifier=classifier,
        classifier_weight=options['classifier_weight'], total_steps=budget.total_steps,
        replay_seed_a=seed+20, replay_seed_b=seed+21)
    result = fit_population(trainer, hand, font, budget, hand_seed=seed+22, font_seed=seed+23,
                            font_sampler=sampler, output_directory=output_directory)
    if any(torch.equal(old, next(network.parameters())) for old, network in zip(initial, networks)):
        raise ValueError('Every trainable GAN network must update')
    if not all(torch.isfinite(p).all() for network in networks for p in network.parameters()):
        raise ValueError('Finite trained GAN weights required')
    for key, value in classifier.state_dict().items():
        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    for entry in result['history']:
        if _sha(Path(output_directory)/entry['generator_file']) != entry['generator_sha256']:
            raise ValueError('Generator checkpoint changed after training')
    final = result['history'][-1]
    return dict(kind='generator', path=str(Path(output_directory)/final['generator_file']),
        sha256=final['generator_sha256'], classifier=classifier_state,
        metrics=dict(epochs=result['completed_epochs'], updates=result['completed_steps'],
                     all_four_networks_updated=True, frozen_classifier_unchanged=True), policy=dict(options))


def predict_b7(state, images):
    if state.get('kind') != 'b7' or state.get('branch') not in ('ood', 'seen'):
        raise ValueError('B7 checkpoint state required')
    tensor = prepare_classifier_images(images, augmentation=None, **state['normalization'])
    model = load_frozen_classifier(GraphemeB7Head(EfficientNet.from_name('efficientnet-b7'), branch=state['branch']),
                                  state['path'], expected_sha256=state['sha256'])
    with torch.no_grad():
        return model(tensor)


def predict_generated(state, images):
    if state.get('kind') != 'generator' or state['classifier'].get('kind') != 'font':
        raise ValueError('Generator and associated font checkpoint required')
    tensor = prepare_handwriting(images)
    generator = load_frozen_classifier(BengaliGenerator(), state['path'], expected_sha256=state['sha256'])
    classifier = load_frozen_classifier(BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0')),
        state['classifier']['path'], expected_sha256=state['classifier']['sha256'])
    with torch.no_grad():
        return classifier(generator(tensor))
