"""Source RGB boundary with the reusable-normalization neural lifecycle."""
import numpy as np
from sciona.adversarial_image_boundary import batches,encode_real_entries
from sciona.decomposed_adversarial_lifecycle import attack
from sciona.adversarial_updates import configuration


def attack_rgb(rgb, *, mode, epsilon, initializations, targets=None, momentum=1.,
               non_targeted_iterations=10):
    configuration(mode,epsilon,non_targeted_iterations=non_targeted_iterations)
    if mode=='targeted' and targets is None:raise ValueError('targeted RGB attack requires target classes')
    if mode=='non_targeted' and targets is not None:raise ValueError('non-targeted attack infers labels')
    images=[];labels=[];pngs=[];batch_losses=[]
    for count,normalized,padded_targets in batches(rgb,targets=targets):
        result=attack(normalized,mode=mode,epsilon=epsilon,initializations=initializations,
                      targets=padded_targets,momentum=momentum,non_targeted_iterations=non_targeted_iterations)
        images.append(result['images'][:count].copy());labels.append(result['labels'][:count].copy())
        pngs.extend(encode_real_entries(result['images'],count));batch_losses.append(result['losses'])
    return {'images':np.concatenate(images),'labels':np.concatenate(labels),'pngs':pngs,
            'batch_losses':batch_losses,'saved_pixel_bound_guaranteed':False}
