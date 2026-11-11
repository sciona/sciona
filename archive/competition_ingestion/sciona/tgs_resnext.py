"""Independent source-topology ResNeXt50 TGS realization in PyTorch.

Pinned source is documented in competition_tgs_source_triage.json. Encoder
normalization and grouped kernels follow the explicit reconstruction contract;
historical Keras checkpoint/runtime equivalence is not established. No weights
are downloaded. Native groups bind channels explicitly, avoiding the reference
Python Lambda loop's late-bound group index.
"""
import torch
from torch import nn
from torch.nn import functional as F

from sciona.tgs_keras_layers import KerasBatchNorm, KerasDecoder


def pointwise(cin, cout, stride=1):
    conv = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
    nn.init.xavier_uniform_(conv.weight)
    return conv


class ResidualGroup(nn.Module):
    def __init__(self, incoming, width, stride, projection):
        super().__init__()
        self.reduce = pointwise(incoming, width)
        self.reduce_norm = KerasBatchNorm(width, epsilon=2e-5)
        self.grouped = nn.Conv2d(width, width, 3, stride=stride, padding=1, groups=32, bias=False)
        # Reference constructs 32 separately initialized narrow convolutions.
        for weight in self.grouped.weight.chunk(32, dim=0):
            nn.init.xavier_uniform_(weight)
        self.group_norm = KerasBatchNorm(width, epsilon=2e-5)
        self.expand = pointwise(width, width * 2)
        self.expand_norm = KerasBatchNorm(width * 2, epsilon=2e-5)
        self.shortcut = nn.Sequential(pointwise(incoming, width * 2, stride),
                                      KerasBatchNorm(width * 2, epsilon=2e-5)) if projection else nn.Identity()

    def forward(self, value):
        skip_activation = self.reduce_norm(self.reduce(value)).relu()
        residual = self.group_norm(self.grouped(skip_activation)).relu()
        residual = self.expand_norm(self.expand(residual))
        return (residual + self.shortcut(value)).relu(), skip_activation


class TGSResNeXt50(nn.Module):
    def __init__(self, *, probabilities=False):
        super().__init__()
        self.probabilities = probabilities
        self.input_norm = KerasBatchNorm(3, epsilon=2e-5, scale=False)
        self.stem_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        nn.init.xavier_uniform_(self.stem_conv.weight)
        self.stem_norm = KerasBatchNorm(64, epsilon=2e-5)
        self.stages = nn.ModuleList()
        incoming = 64
        for stage, repetitions in enumerate((3, 4, 6, 3)):
            width = 128 * 2 ** stage
            blocks = nn.ModuleList()
            for index in range(repetitions):
                blocks.append(ResidualGroup(incoming, width, 2 if stage and index == 0 else 1, index == 0))
                incoming = width * 2
            self.stages.append(blocks)
        self.decoders = nn.ModuleList()
        for skip, outgoing in zip((1024, 512, 256, 64, 0), (128, 64, 32, 16, 8)):
            self.decoders.append(KerasDecoder(incoming, skip, outgoing))
            incoming = outgoing
        self.dropout = nn.Dropout2d(.2)
        self.prediction = nn.Conv2d(8, 1, 1)
        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)

    def forward(self, images):
        if not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or images.device.type != 'cpu':
            raise ValueError('float32 CPU tensor required')
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1:] != (3, 224, 224) or not torch.isfinite(images).all():
            raise ValueError('finite source-size N3x224x224 input required')
        stem = self.stem_norm(self.stem_conv(self.input_norm(images))).relu()
        # Reference explicitly zero pads before valid max pooling.
        value = F.max_pool2d(F.pad(stem, (1, 1, 1, 1)), 3, stride=2)
        skips = [stem]
        for stage, blocks in enumerate(self.stages):
            for index, block in enumerate(blocks):
                value, intermediate = block(value)
                if stage > 0 and index == 0:
                    skips.append(intermediate)
        skips.reverse()
        for index, decoder in enumerate(self.decoders):
            value = decoder(value, skips[index] if index < len(skips) else None)
        value = self.prediction(self.dropout(value))
        return value.sigmoid() if self.probabilities else value
