"""Source attack recurrence with a graph-injected reusable row operation.

Derived from the reviewed Apache-2.0 attack adapters. See
 docs/licenses/Adversarial-*-Apache-2.0.txt. The graph supplies normalization;
this module contains source recipe policy and no fallback numerical provider.
"""
import base64
import math
import numpy as np
import torch
from sciona.adversarial_ensemble import build_ensemble, gradient
from sciona.adversarial_updates import configuration
from sciona.adversarial_image_boundary import batches, encode_real_entries
from sciona.adversarial_runtime import keys
from sciona.dfdc_codec import encode_array


def update(x, gradient, accumulated, lower, upper, *, normalizer, mode, epsilon, momentum=1.,
           non_targeted_iterations=10):
    """Return projected image and next momentum, without mutating input arrays.

    Values are NHWC float32/64 in normalized [-1,1]. Zero normalization
    denominators explicitly fail instead of silently propagating source NaNs;
    no stabilizing epsilon or substituted update is introduced.
    """
    config=configuration(mode,epsilon,non_targeted_iterations=non_targeted_iterations)
    if type(momentum) not in (float,int) or not math.isfinite(momentum) or momentum<0:
        raise ValueError('momentum must be finite and nonnegative')
    arrays=(x,gradient,accumulated,lower,upper)
    if (not isinstance(x,np.ndarray) or x.ndim!=4 or min(x.shape)==0 or x.shape[-1]!=3
            or x.dtype not in (np.dtype('float32'),np.dtype('float64'))
            or any(not isinstance(a,np.ndarray) or a.shape!=x.shape or a.dtype!=x.dtype
                   or not np.isfinite(a).all() for a in arrays)):
        raise ValueError('update requires matching finite NHWC floating arrays')
    if np.any(lower < -1) or np.any(upper > 1) or np.any(lower>x) or np.any(x>upper):
        raise ValueError('image must lie inside valid normalized clipping bounds')
    # The neural boundary supplies contiguous NHWC arrays. Restrict this
    # adapter to that qualified layout rather than changing reduction order.
    if any(not a.flags.c_contiguous for a in arrays):
        raise ValueError('contiguous NHWC arrays required')
    statistic='mean_absolute' if mode=='non_targeted' else 'population_std'
    scaled=normalizer(gradient.reshape(len(x),-1),statistic).reshape(x.shape)
    noise=momentum*accumulated+scaled
    if mode=='targeted':
        noise=normalizer(noise.reshape(len(x),-1),'population_std').reshape(x.shape)
        result=x-config['alpha']*np.clip(np.round(noise),-2,2)
    else:
        result=x+config['alpha']*np.sign(noise)
    result=np.clip(result,lower,upper)
    if not np.isfinite(result).all() or not np.isfinite(noise).all():
        raise ValueError('update became nonfinite')
    return result,noise


def attack(images, *, normalizer, mode, epsilon, initializations, targets=None, momentum=1.,
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
        x,noise=update(x,raw,noise,lower,upper,normalizer=normalizer,mode=mode,epsilon=epsilon,
                       momentum=momentum,non_targeted_iterations=non_targeted_iterations)
        losses.append(result['loss'])
    return {'images':x,'momentum':noise,'labels':labels.numpy().copy(),
            'losses':losses,'configuration':config,
            'initialization_kinds':{scope:spec['kind'] for scope,spec in initializations.items()}}


def attack_rgb(rgb, *, normalizer, mode, epsilon, initializations, targets=None, momentum=1.,
               non_targeted_iterations=10):
    configuration(mode,epsilon,non_targeted_iterations=non_targeted_iterations)
    if mode=='targeted' and targets is None:raise ValueError('targeted RGB attack requires target classes')
    if mode=='non_targeted' and targets is not None:raise ValueError('non-targeted attack infers labels')
    images=[];labels=[];pngs=[];batch_losses=[]
    for count,normalized,padded_targets in batches(rgb,targets=targets):
        result=attack(normalized,normalizer=normalizer,mode=mode,epsilon=epsilon,initializations=initializations,
                      targets=padded_targets,momentum=momentum,non_targeted_iterations=non_targeted_iterations)
        images.append(result['images'][:count].copy());labels.append(result['labels'][:count].copy())
        pngs.extend(encode_real_entries(result['images'],count));batch_losses.append(result['losses'])
    return {'images':np.concatenate(images),'labels':np.concatenate(labels),'pngs':pngs,
            'batch_losses':batch_losses,'saved_pixel_bound_guaranteed':False}


def execute(prepared, normalizer):
    if not callable(normalizer):raise ValueError("Explicit row-normalization callable required")
    keys(prepared,('rgb','targets','config','initializations'))
    result=attack_rgb(prepared['rgb'],normalizer=normalizer,targets=prepared['targets'],
                      initializations=prepared['initializations'],**prepared['config'])
    return {'version':1,'normalized_images':encode_array(result['images']),
            'labels':encode_array(result['labels']),
            'pngs':[base64.b64encode(data).decode('ascii') for data in result['pngs']],
            'batch_losses':result['batch_losses'],
            'saved_pixel_bound_guaranteed':False,
            'initialization_kinds':{s:v['kind'] for s,v in prepared['initializations'].items()}}

