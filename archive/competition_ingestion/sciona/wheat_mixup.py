"""Source batch mixup after the 20-epoch population warmup."""
import numpy as np
import torch


def mixup_batch(images, targets, epoch, detector, *, python_rng, numpy_rng, torch_generator):
    if detector not in ('effdet', 'fasterrcnn') or type(epoch) is not int or not 0 <= epoch < 100:
        raise ValueError('source detector and epoch required')
    if not len(images) or len(images) != len(targets):
        raise ValueError('nonempty aligned image and target batches required')
    apply = python_rng.random() > .5 and epoch >= 20
    stacked = torch.stack(list(images))
    if not apply:
        return stacked, targets
    permutation = torch.randperm(len(images), generator=torch_generator)
    weight = np.clip(numpy_rng.beta(1., 1.), .35, .65)
    mixed = weight * stacked + (1-weight) * stacked[permutation]
    fields = ('boxes', 'labels') if detector == 'effdet' else ('boxes', 'labels', 'area', 'iscrowd')
    output_targets = []
    for index, other in enumerate(permutation.tolist()):
        output_targets.append(targets[index] if index == other else
            {key:torch.cat([targets[index][key], targets[other][key]]) for key in fields})
    return mixed, output_targets
