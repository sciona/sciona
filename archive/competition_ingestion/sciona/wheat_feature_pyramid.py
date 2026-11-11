"""Winner's four-level feature pyramid, including historical initialization."""
from collections import OrderedDict

from torch import nn
from torch.nn import functional as F


class WheatFeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for channels in (256, 512, 1024, 2048):
            # torchvision 0.5 iterates children (ModuleLists), so its later
            # Conv2d initialization loop performs no reinitialization.
            self.inner_blocks.append(nn.Conv2d(channels, 256, 1))
            self.layer_blocks.append(nn.Conv2d(256, 256, 3, padding=1))

    def forward(self, features):
        names, values = list(features), list(features.values())
        if len(values) != 4:
            raise ValueError('four ResNet feature levels required')
        inner = self.inner_blocks[-1](values[-1])
        outputs = [self.layer_blocks[-1](inner)]
        for level in (2, 1, 0):
            lateral = self.inner_blocks[level](values[level])
            inner = lateral + F.interpolate(inner, size=lateral.shape[-2:], mode='nearest')
            outputs.insert(0, self.layer_blocks[level](inner))
        names.append('pool')
        outputs.append(F.max_pool2d(outputs[-1], 1, 2, 0))
        return OrderedDict(zip(names, outputs))
