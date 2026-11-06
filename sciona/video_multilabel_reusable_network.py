"""Video model policy using independently reusable differentiable operations."""
import torch
from sciona.video_multilabel_network import VideoHead
from sciona.atoms.ml.masked_attention import masked_attention_pool, sigmoid_context_gate


class ReusableVideoHead(VideoHead):
    def __init__(self, features, hidden, labels, *,
                 pool_function=masked_attention_pool, gate_function=sigmoid_context_gate):
        super().__init__(features, hidden, labels)
        self.pool_function = pool_function
        self.gate_function = gate_function

    def pool(self, frames, mask):
        if (frames.ndim != 3 or frames.shape[2] != self.features
                or mask.shape != frames.shape[:2] or mask.dtype != torch.bool
                or not frames.is_floating_point()):
            raise ValueError('Expected batch/time/feature tensor and boolean mask')
        # Mask before the learned score model as well, keeping its gradients
        # independent of nonfinite padding supplied by callers.
        clean = torch.where(mask[:, :, None], frames, torch.zeros_like(frames))
        logits = self.attention(clean).squeeze(-1)
        return self.pool_function(frames, logits, mask)

    def forward(self, frames, mask):
        pooled, _ = self.pool(frames, mask)
        return self.classifier(self.gate_function(pooled, self.context(pooled)))
