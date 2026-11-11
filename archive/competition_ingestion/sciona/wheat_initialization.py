"""Pinned source pretrained detectors with fresh one-class classification towers."""
import hashlib
import io
from pathlib import Path

import torch


CHECKPOINTS = {
    'ed5': 'ef44aea85cc8780edd1be6a260bd2f31cbef581b5748433e691aa61540867220',
    'ed6': '51cb0132a9c30e0aab1eb169f1881d3fccbcf78a71c1c9349af8eb04a181498b',
    'ed7': 'f05bf7142a08e3ed2fbd0202d6ecc79e886118aee1b5ad9ccb764933b14a0a12',
}


def initialized_detector(backbone, checkpoint, *, image_size, mode='train'):
    if backbone not in CHECKPOINTS or mode not in ('train', 'eval'):
        raise ValueError('source detector variant and train/eval bench required')
    if type(image_size) is not int or image_size < 1:
        raise ValueError('positive image size required')
    raw = Path(checkpoint).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKPOINTS[backbone]:
        raise ValueError('publisher detector checkpoint differs')
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    from effdet import EfficientDet, get_efficientdet_config, DetBenchTrain, DetBenchEval
    from effdet.efficientdet import HeadNet
    config = get_efficientdet_config('tf_efficientdet_d' + backbone[-1])
    model = EfficientDet(config, pretrained_backbone=False)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('pretrained detector tensor inventory differs')
    for key, value in state.items():
        target = expected[key]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != target.shape or value.dtype != target.dtype
                or not torch.isfinite(value).all()):
            raise ValueError('pretrained detector tensor contract differs')
    model.load_state_dict(state, strict=True)
    config.num_classes = 1
    config.image_size = image_size
    model.class_net = HeadNet(config, num_outputs=1, norm_kwargs=dict(eps=.001, momentum=.01))
    return (DetBenchTrain if mode == 'train' else DetBenchEval)(model, config)
