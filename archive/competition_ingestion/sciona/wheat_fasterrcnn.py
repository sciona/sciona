"""Source-configured ResNet-152 Faster R-CNN for Global Wheat execution."""
import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from sciona.wheat_resnet_backbone import initialized_resnet152
from sciona.wheat_feature_pyramid import WheatFeaturePyramid
from sciona.wheat_proposals import WheatRegionProposalNetwork
from sciona.wheat_detection_losses import roi_loss
from sciona.wheat_image_transform import WheatImageTransform


class WheatBackbone(nn.Module):
    out_channels = 256

    def __init__(self, checkpoint):
        super().__init__()
        self.body = IntermediateLayerGetter(initialized_resnet152(checkpoint),
            return_layers={f'layer{i}': str(i-1) for i in range(1, 5)})
        self.fpn = WheatFeaturePyramid()

    def forward(self, images):
        return self.fpn(self.body(images))


class WheatRoIHeads(RoIHeads):
    def __init__(self):
        super().__init__(MultiScaleRoIAlign(['0', '1', '2', '3'], 7, 2),
                         TwoMLPHead(256 * 7 * 7, 1024), FastRCNNPredictor(1024, 91),
                         .5, .5, 512, .25, None, .05, .5, 100)

    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for target in targets:
                if target['boxes'].dtype not in (torch.float16, torch.float32, torch.float64):
                    raise ValueError('floating-point target boxes required')
                if target['labels'].dtype != torch.int64:
                    raise ValueError('int64 target labels required')
        if self.training:
            if targets is None or any(target['boxes'].numel() == 0 for target in targets):
                raise ValueError('historical ROI training requires nonempty ground truth')
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        if sum(len(value) for value in proposals) == 0:
            raise ValueError('historical box decoder requires nonempty total proposals')
        pooled = self.box_roi_pool(features, proposals, image_shapes)
        logits, deltas = self.box_predictor(self.box_head(pooled))
        if self.training:
            classification, regression = roi_loss(logits, deltas, labels, regression_targets)
            return [], dict(loss_classifier=classification, loss_box_reg=regression)
        boxes, scores, labels = self.postprocess_detections(logits, deltas, proposals, image_shapes)
        return [dict(boxes=b, scores=s, labels=l) for b, s, l in zip(boxes, scores, labels)], {}


def initialized_fasterrcnn(checkpoint):
    backbone = WheatBackbone(checkpoint)
    rpn = WheatRegionProposalNetwork()
    roi = WheatRoIHeads()
    model = GeneralizedRCNN(backbone, rpn, roi, WheatImageTransform())
    # The source constructs a 91-class detector, then replaces its predictor.
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    return model
