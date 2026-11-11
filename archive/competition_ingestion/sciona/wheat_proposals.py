"""Source proposal filtering and losses on the installed RPN execution shell."""
import torch
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import boxes as box_ops

from sciona.wheat_anchors import WheatAnchorGenerator
from sciona.wheat_detection_losses import rpn_loss


class WheatRegionProposalNetwork(RegionProposalNetwork):
    def __init__(self):
        super().__init__(WheatAnchorGenerator(), RPNHead(256, 3), .7, .3, 256, .5,
                         {'training': 2000, 'testing': 1000},
                         {'training': 2000, 'testing': 1000}, .7)

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        return rpn_loss(objectness, pred_bbox_deltas, labels, regression_targets, self.fg_bg_sampler)

    def assign_targets_to_anchors(self, anchors, targets):
        if any(target['boxes'].numel() == 0 for target in targets):
            raise ValueError('historical RPN requires a nonempty ground-truth box set')
        return super().assign_targets_to_anchors(anchors, targets)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        scores = objectness.detach().reshape(proposals.shape[0], -1)
        levels = torch.cat([torch.full((count,), level, dtype=torch.int64, device=proposals.device)
                            for level, count in enumerate(num_anchors_per_level)])
        levels = levels[None, :].expand_as(scores)
        selected = self._get_top_n_idx(scores, num_anchors_per_level)
        batch = torch.arange(proposals.shape[0], device=proposals.device)[:, None]
        boxes, scores, levels = proposals[batch, selected], scores[batch, selected], levels[batch, selected]
        result_boxes, result_scores = [], []
        for image_boxes, image_scores, image_levels, shape in zip(boxes, scores, levels, image_shapes):
            image_boxes = box_ops.clip_boxes_to_image(image_boxes, shape)
            keep = box_ops.remove_small_boxes(image_boxes, self.min_size)
            image_boxes, image_scores, image_levels = image_boxes[keep], image_scores[keep], image_levels[keep]
            keep = box_ops.batched_nms(image_boxes, image_scores, image_levels, self.nms_thresh)[:self.post_nms_top_n()]
            result_boxes.append(image_boxes[keep])
            result_scores.append(image_scores[keep])
        return result_boxes, result_scores
