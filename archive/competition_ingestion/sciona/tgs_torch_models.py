"""Independent topology reconstruction of the three TGS PyTorch variants.

Mathematical reference: phalanx/unet_model.py at
2f81d4dd8d50a01579e5f7650259dde92c5c3b8d. Uses library ResNet34 without
downloading weights. Pretrained initialization/provenance belongs to the full
training lifecycle. Variant 5 dropout is corrected to respect evaluation mode.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34


def resize(value, shape):
    return F.interpolate(value, size=shape, mode='bilinear', align_corners=True)


def conv_norm_elu(cin, cout, kernel=3, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ELU())


class Excitation(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.spatial = nn.Conv2d(channels, 1, 1, bias=False)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // reduction, 1),
                                     nn.ReLU(), nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid())

    def forward(self, value):
        return value * (self.spatial(value).sigmoid() + self.channel(value))


class Pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.ModuleList([conv_norm_elu(512, 512, k, 2) for k in (5, 3)])
        self.project = nn.ModuleList([conv_norm_elu(512, 256, k) for k in (5, 3)])
        self.local = conv_norm_elu(512, 256, 1)
        self.global_projection = nn.Conv2d(512, 256, 1, bias=False)

    def forward(self, value):
        coarse = self.down[0](value)
        finer = self.project[0](coarse)
        attention = finer + resize(self.project[1](self.down[1](coarse)), finer.shape[-2:])
        pooled = self.global_projection(F.adaptive_avg_pool2d(value, 1))
        return self.local(value) * resize(attention, value.shape[-2:]) + resize(pooled, value.shape[-2:])


class SkipDecoder(nn.Module):
    def __init__(self, incoming, skip):
        super().__init__()
        self.up = nn.ConvTranspose2d(incoming, 32, 2, stride=2)
        self.skip = nn.Conv2d(skip, 32, 1, bias=False)
        self.fuse = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(), Excitation(64, 16))

    def forward(self, value, skip):
        return self.fuse(torch.cat((self.up(value), self.skip(skip)), dim=1))


class TGSResNet34(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if isinstance(variant, bool) or variant not in (3, 4, 5):
            raise ValueError('variant must be 3, 4 or 5')
        self.variant = variant
        self.input_size = 128 if variant == 5 else 256
        backbone = resnet34(weights=None)
        stem_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False) if variant == 5 else backbone.conv1
        self.stem = nn.Sequential(stem_conv, backbone.bn1, backbone.relu)
        self.encoder = nn.ModuleList([nn.Sequential(block, Excitation(width, 4)) for block, width in
                                     zip((backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4), (64, 128, 256, 512))])
        self.center = nn.Sequential(Pyramid(), nn.MaxPool2d(2))
        self.decoder = nn.ModuleList([SkipDecoder(cin, skip) for cin, skip in
                                     ((256, 512), (64, 256), (64, 128), (64, 64))])
        if variant != 5:
            self.final_up = nn.Sequential(conv_norm_elu(64, 32), conv_norm_elu(32, 64), Excitation(64, 16))
        self.dropout = nn.Dropout2d(.4)
        if variant == 3:
            self.pixel_features = conv_norm_elu(320, 64)
            self.pixel_head = nn.Conv2d(64, 1, 1, bias=False)
            self.image_features = nn.Sequential(nn.Dropout(.4), nn.Linear(512, 64), nn.ELU())
            self.image_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
            self.head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.ELU(), nn.Conv2d(64, 1, 1, bias=False))
        else:
            channels, hidden = (256, 32) if variant == 5 else (320, 64)
            self.head = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.ELU(), nn.Conv2d(hidden, 1, 1, bias=False))

    def forward(self, images):
        if not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or images.device.type != 'cpu':
            raise ValueError('float32 CPU input required')
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1:] != (3, self.input_size, self.input_size) or not torch.isfinite(images).all():
            raise ValueError('finite input with source variant spatial dimensions required')
        value = self.stem(images)
        encoded = []
        for block in self.encoder:
            value = block(value)
            encoded.append(value)
        value = self.center(value)
        decoded = []
        for block, skip in zip(self.decoder, reversed(encoded)):
            value = block(value, skip)
            decoded.append(value)
        if self.variant != 5:
            value = self.final_up(resize(value, images.shape[-2:]))
            decoded.append(value)
        hypercolumns = torch.cat([resize(item, images.shape[-2:]) for item in reversed(decoded)], dim=1)
        if self.variant != 4:
            hypercolumns = self.dropout(hypercolumns)
        if self.variant != 3:
            return self.head(hypercolumns)
        pixels = self.pixel_features(hypercolumns)
        pooled = F.adaptive_avg_pool2d(encoded[-1], 1).flatten(1)
        image_features = self.image_features(pooled)
        fused = torch.cat((pixels, resize(image_features[:, :, None, None], images.shape[-2:])), dim=1)
        return self.head(fused), self.pixel_head(pixels), self.image_head(image_features).flatten()
