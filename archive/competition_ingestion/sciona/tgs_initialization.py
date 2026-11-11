"""Strict mapping of authenticated historical ResNet34 weights to TGS variants."""
import hashlib
from pathlib import Path
import torch


def load_resnet34_reference(model, path, sha256):
    """Map the encoder only; decoder/attention initialization stays independent.

    Requires caller-qualified full SHA256 plus the historical publisher hash
    prefix. A restricted data-only reader handles the historical tar format.
    Variant5's replaced input conv
    is intentionally excluded; reference batchnorm buffers still load.
    """
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256 or digest != '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef':
        raise ValueError('historical ResNet34 reference digest mismatch')
    from sciona.tgs_legacy_weights import read_legacy_resnet34
    source = read_legacy_resnet34(data)
    if not isinstance(source, dict) or not all(isinstance(v, torch.Tensor) for v in source.values()):
        raise ValueError('tensor-only reference state required')
    current = model.state_dict()
    updates, consumed = {}, set()
    for name, target in current.items():
        origin = None
        if name == 'stem.0.weight' and model.variant != 5:
            origin = 'conv1.weight'
        elif name.startswith('stem.1.'):
            origin = 'bn1.' + name.removeprefix('stem.1.')
        elif name.startswith('encoder.'):
            parts = name.split('.')
            if parts[2] == '0':
                origin = 'layer' + str(int(parts[1]) + 1) + '.' + '.'.join(parts[3:])
        if origin is None:
            continue
        if name.endswith('num_batches_tracked') and origin not in source:
            updates[name] = torch.zeros_like(target)
            continue
        if origin not in source or source[origin].shape != target.shape or source[origin].dtype != target.dtype or not torch.isfinite(source[origin]).all():
            raise ValueError('reference encoder contract mismatch: ' + str(origin))
        updates[name] = source[origin].clone()
        consumed.add(origin)
    excluded = {'fc.weight', 'fc.bias'}
    if model.variant == 5:
        excluded.add('conv1.weight')
    if set(source) - consumed != excluded:
        raise ValueError('unexpected unused reference tensors')
    if not updates:
        raise ValueError('no encoder tensors mapped')
    # All validation precedes the only model mutation.
    current.update(updates)
    model.load_state_dict(current, strict=True)
    return dict(reference_sha256=digest, source_tensors_consumed=len(consumed),
                model_tensors_initialized=len(updates), variant=model.variant)


def load_resnext50_reference(model, path, sha256):
    """Map authenticated no-top Keras HWIO tensors to native grouped OIHW.

    All 764 source tensors must be consumed; decoder parameters stay untouched.
    This establishes initialization mapping, not historical runtime equivalence.
    """
    import io
    import h5py
    import numpy as np
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if (digest != sha256 or digest != '3bcb9dedb226c5e5cdd3510d25cdc33297f59016a6d7069758024caa13e3172d'
            or hashlib.md5(data).hexdigest() != '7ade5c8aac9194af79b1724229bdaa50'):
        raise ValueError('historical ResNeXt50 reference digest mismatch')
    current = model.state_dict()
    updates, consumed = {}, set()
    with h5py.File(io.BytesIO(data), 'r') as source:
        datasets = set()
        source.visititems(lambda name, obj: datasets.add(name) if isinstance(obj, h5py.Dataset) else None)
        def read(layer, field):
            key = f'{layer}/{layer}/{field}:0'
            value = source[key][()]
            if value.dtype != np.float32 or not np.isfinite(value).all() or key in consumed:
                raise ValueError('invalid reference tensor: ' + key)
            consumed.add(key)
            return torch.from_numpy(value.copy())
        def put(name, value):
            if name not in current or value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError('reference target mismatch: ' + name)
            updates[name] = value.contiguous()
        def norm(target, layer, scale=True):
            for field in ('gamma', 'beta', 'moving_mean', 'moving_variance'):
                value = read(layer, field) if field != 'gamma' or scale else torch.ones_like(current[target + '.gamma'])
                if field == 'moving_variance' and (value < 0).any():
                    raise ValueError('negative reference moving variance')
                put(target + '.' + field, value)
        def conv(target, layer):
            put(target + '.weight', read(layer, 'kernel').permute(3, 2, 0, 1))
        norm('input_norm', 'bn_data', scale=False)
        norm('stem_norm', 'bn0')
        conv('stem_conv', 'conv0')
        for stage, blocks in enumerate(model.stages):
            for index, _ in enumerate(blocks):
                target = f'stages.{stage}.{index}'
                layer = f'stage{stage + 1}_unit{index + 1}'
                conv(target + '.reduce', layer + '_conv1')
                conv(target + '.expand', layer + '_conv3')
                for target_norm, origin in [('reduce_norm', 'bn1'), ('group_norm', 'bn2'), ('expand_norm', 'bn3')]:
                    norm(target + '.' + target_norm, layer + '_' + origin)
                groups = [read(layer + f'_conv2_{group}', 'kernel').permute(3, 2, 0, 1) for group in range(32)]
                put(target + '.grouped.weight', torch.cat(groups, dim=0))
                if index == 0:
                    conv(target + '.shortcut.0', layer + '_sc')
                    norm(target + '.shortcut.1', layer + '_sc_bn')
        expected = {key for key in current if key.startswith(('input_norm.', 'stem_norm.', 'stem_conv.', 'stages.'))}
        if datasets != consumed or len(consumed) != 764 or set(updates) != expected:
            raise ValueError('incomplete reference encoder mapping')
    current.update(updates)
    model.load_state_dict(current, strict=True)
    return dict(reference_sha256=digest, source_tensors_consumed=len(consumed),
                model_tensors_initialized=len(updates), architecture='resnext50')
