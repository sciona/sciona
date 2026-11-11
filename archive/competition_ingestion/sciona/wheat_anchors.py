"""Historical float32 grid shifts, including mixed-precision feature inputs."""
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator


class WheatAnchorGenerator(AnchorGenerator):
    def __init__(self):
        super().__init__(((32,), (64,), (128,), (256,), (512,)), ((.5, 1., 2.),) * 5)
        self.cell_anchors = None

    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            return
        # Historical construction rounds base anchors in the feature dtype;
        # casting anchors initially constructed in float32 changes half values.
        self.cell_anchors = [self.generate_anchors(sizes, ratios, dtype, device)
                             for sizes, ratios in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        if len(grid_sizes) != 5 or len(strides) != 5 or len(self.cell_anchors) != 5:
            raise ValueError('five source feature levels required')
        result = []
        for (height, width), (stride_y, stride_x), base in zip(grid_sizes, strides, self.cell_anchors):
            x = torch.arange(width, dtype=torch.float32, device=base.device) * stride_x
            y = torch.arange(height, dtype=torch.float32, device=base.device) * stride_y
            y, x = torch.meshgrid(y, x, indexing='ij')
            shifts = torch.stack([x.flatten(), y.flatten(), x.flatten(), y.flatten()], dim=1)
            result.append((shifts[:, None, :] + base[None, :, :]).reshape(-1, 4))
        return result
