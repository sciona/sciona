"""Initialize the winner's ResNet-152 backbone from its publisher artifact."""
import hashlib
from pathlib import Path

import torch
from torchvision.models import resnet152

from sciona.wheat_frozen_normalization import source_frozen_batch_norm
from sciona.wheat_legacy_weights import read_legacy_resnet152


CHECKPOINT_SHA256 = 'b121ed2db97ec7e9f55a91300ceaf85a326de955e8a4ae09e3a0c8170d27f14f'


def initialized_resnet152(checkpoint):
    raw = Path(checkpoint).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('publisher ResNet-152 checkpoint differs')
    state = read_legacy_resnet152(raw)
    model = resnet152(weights=None, norm_layer=source_frozen_batch_norm)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('pretrained backbone tensor inventory differs')
    for key, value in state.items():
        target = expected[key]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != target.shape or value.dtype != target.dtype
                or not torch.isfinite(value).all()):
            raise ValueError('pretrained backbone tensor contract differs')
    model.load_state_dict(state, strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(stage in name for stage in ('layer2', 'layer3', 'layer4')))
    return model
