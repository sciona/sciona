"""Regression objectives used by the winner's torchvision 0.5 detector."""
import torch
from torch.nn import functional as F


def rpn_loss(objectness, box_deltas, labels, targets, sampler):
    positive, negative = sampler(labels)
    positive = torch.nonzero(torch.cat(positive)).squeeze(1)
    negative = torch.nonzero(torch.cat(negative)).squeeze(1)
    selected = torch.cat([positive, negative])
    all_labels, all_targets = torch.cat(labels), torch.cat(targets)
    regression = F.l1_loss(box_deltas[positive], all_targets[positive], reduction='sum') / selected.numel()
    classification = F.binary_cross_entropy_with_logits(objectness.flatten()[selected], all_labels[selected])
    return classification, regression


def roi_loss(class_logits, box_deltas, labels, targets):
    all_labels, all_targets = torch.cat(labels), torch.cat(targets)
    positive = torch.nonzero(all_labels > 0).squeeze(1)
    boxes = box_deltas.reshape(class_logits.shape[0], -1, 4)
    regression = F.smooth_l1_loss(boxes[positive, all_labels[positive]], all_targets[positive],
                                  beta=1.0, reduction='sum') / all_labels.numel()
    return F.cross_entropy(class_logits, all_labels), regression
