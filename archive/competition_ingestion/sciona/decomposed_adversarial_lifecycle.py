"""Complete neural schedules using the reusable normalization composition.

Input/output boundary is NHWC CPUfloat32, not filesystem image decoding. Models
are explicit random or caller-supplied state; no implicit pretrained download.
"""
import math
import numpy as np
import torch
from sciona.adversarial_ensemble import build_ensemble,gradient
from sciona.adversarial_updates import configuration
from sciona.decomposed_adversarial_updates import update


def attack(images, *, mode, epsilon, initializations, targets=None, momentum=1.,
           non_targeted_iterations=10):
    config=configuration(mode,epsilon,non_targeted_iterations=non_targeted_iterations)
    if (not isinstance(images,np.ndarray) or images.dtype!=np.float32 or images.ndim!=4
            or images.shape[1:]!=(299,299,3) or images.shape[0]<1
            or not np.isfinite(images).all() or np.any(images < -1) or np.any(images > 1)):
        raise ValueError('finite normalized NHWCfloat32 full-resolution batch required')
    if type(momentum) not in (int,float) or not math.isfinite(momentum) or momentum<0:
        raise ValueError('finite nonnegative momentum required')
    if mode=='targeted':
        if (not isinstance(targets,np.ndarray) or targets.dtype!=np.int64 or targets.shape!=(len(images),)
                or np.any(targets<0) or np.any(targets>=1001)):
            raise ValueError('explicit per-example int64 target classes required')
        labels=torch.from_numpy(targets.copy())
    else:
        if targets is not None:raise ValueError('non-targeted labels must be inferred')
        labels=None
    models=build_ensemble(config['branch'],initializations)
    x=images.copy();noise=np.zeros_like(x)
    bound=2.*epsilon/255.
    lower=np.maximum(images-bound,-1);upper=np.minimum(images+bound,1)
    losses=[];first_labels=None
    for iteration in range(config['iterations']):
        result=gradient(models,torch.from_numpy(x).permute(0,3,1,2),
                        branch=config['branch'],iteration=iteration,labels=labels)
        labels=result['labels']
        if first_labels is None:first_labels=labels.clone()
        elif not torch.equal(labels,first_labels):raise ValueError('attack labels changed after initialization')
        raw=result['gradient'].permute(0,2,3,1).contiguous().numpy()
        x,noise=update(x,raw,noise,lower,upper,mode=mode,epsilon=epsilon,
                       momentum=momentum,non_targeted_iterations=non_targeted_iterations)
        losses.append(result['loss'])
    return {'images':x,'momentum':noise,'labels':labels.numpy().copy(),
            'losses':losses,'configuration':config,
            'initialization_kinds':{scope:spec['kind'] for scope,spec in initializations.items()}}
