"""Independent PyTorch realization of TGS Keras normalization/decoder math.

Source configuration pinned in competition_tgs_source_triage.json. This is a
runtime reconstruction, not historical Keras execution or checkpoint parity.
"""
import torch
from torch import nn


class KerasBatchNorm(nn.Module):
    """Population normalization and Keras2.2.0 fused-backend moving variance.

    TensorFlow1.9 returns Bessel-corrected batch variance; this Keras release
    applies a second sample-size correction before its moving-average update.
    Preserve both source factors, even though later releases changed this.
    """
    def __init__(self, channels, *, epsilon=1e-3, retention=.99, scale=True):
        super().__init__()
        self.epsilon, self.retention = epsilon, retention
        if scale:
            self.gamma = nn.Parameter(torch.ones(channels))
        else:
            self.register_buffer('gamma', torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.register_buffer('moving_mean', torch.zeros(channels))
        self.register_buffer('moving_variance', torch.ones(channels))

    def forward(self, value):
        if value.ndim != 4 or value.shape[1] != len(self.beta):
            raise ValueError('aligned NCHW tensor required')
        if self.training:
            variance, mean = torch.var_mean(value, dim=(0, 2, 3), correction=0)
            count = value.numel() // value.shape[1]
            moving_variance = variance * (count / max(count - 1, 1))
            moving_variance = moving_variance * (count / (count - (1.0 + self.epsilon)))
            with torch.no_grad():
                self.moving_mean.lerp_(mean, 1 - self.retention)
                self.moving_variance.lerp_(moving_variance, 1 - self.retention)
        else:
            mean, variance = self.moving_mean, self.moving_variance
        shape = (1, -1, 1, 1)
        return (value - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + self.epsilon) * self.gamma.reshape(shape) + self.beta.reshape(shape)


class DecoderAttention(nn.Module):
    """Source decoder uses C spatial gates and a C/2 channel bottleneck."""
    def __init__(self, channels):
        super().__init__()
        self.local_gate = nn.Conv2d(channels, channels, 1)
        self.global_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.Linear(channels, channels // 2), nn.ReLU(),
                                         nn.Linear(channels // 2, channels), nn.Sigmoid())
        nn.init.kaiming_normal_(self.local_gate.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.local_gate.bias)
        for layer in self.global_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, value):
        return value * (self.local_gate(value).sigmoid() + self.global_gate(value)[:, :, None, None])


class KerasDecoder(nn.Module):
    """Transpose-4x4, normalization, skip concatenation, conv-3x3, attention."""
    def __init__(self, incoming, skip_channels, outgoing):
        super().__init__()
        self.up = nn.ConvTranspose2d(incoming, outgoing, 4, stride=2, padding=1, bias=False)
        self.up_norm = KerasBatchNorm(outgoing)
        self.conv = nn.Conv2d(outgoing + skip_channels, outgoing, 3, padding=1, bias=False)
        self.norm = KerasBatchNorm(outgoing)
        self.attention = DecoderAttention(outgoing)
        self.skip_channels = skip_channels
        for layer in (self.up, self.conv):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, value, skip=None):
        value = self.up_norm(self.up(value)).relu()
        if self.skip_channels:
            if skip is None or skip.shape[1] != self.skip_channels or skip.shape[0] != value.shape[0] or skip.shape[-2:] != value.shape[-2:]:
                raise ValueError('aligned source skip activation required')
            value = torch.cat((value, skip), dim=1)
        elif skip is not None:
            raise ValueError('final decoder has no skip')
        return self.attention(self.norm(self.conv(value)).relu())
