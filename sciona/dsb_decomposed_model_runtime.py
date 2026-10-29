"""Explicit model-state boundaries for the candidate DSB execution graph.

Inputs are metadata-free epoch/state_dict checkpoint bytes. No weight download,
implicit random predictor, data loading, or checkpoint persistence occurs here.
"""
import math

import torch

from sciona.dsb_checkpoint import restore_checkpoint
from sciona.dsb_components import _integer
from sciona.dsb_network import DetectorNet
from sciona.dsb_decomposed_network import CaseNet, shared_training_models


def restore_training_runtime(classifier_checkpoint, *, topk=5, start_epoch=0,
                             weight_decay=1e-4):
    """Restore shared models and fresh source-style SGD optimizers on CPU.

    This is the source's model warm restart. Momentum and RNG are not restored.
    The epoch runner assigns each pass's learning rate before optimizer use.
    """
    topk=_integer(topk,'topk')
    if not math.isfinite(weight_decay) or weight_decay<0:
        raise ValueError('weight_decay must be finite and nonnegative')
    detector,classifier=shared_training_models(topk)
    epoch=restore_checkpoint(classifier,classifier_checkpoint,start_epoch)
    optimizers=[torch.optim.SGD(model.parameters(),lr=0.,momentum=.9,
                                weight_decay=weight_decay)
                for model in [detector,classifier]]
    return dict(detector=detector,classifier=classifier,
                detector_optimizer=optimizers[0],classifier_optimizer=optimizers[1],
                start_epoch=epoch)


def restore_inference_runtime(detector_checkpoint,classifier_checkpoint,*,topk=5):
    """Restore both explicit states into separate eval-mode inference networks.

    DetectorNet returns predictions; the classifier's internal Net returns
    features and predictions. Neither state is silently substituted for the other.
    Caller chooses checkpoints and verifies their training provenance separately.
    """
    topk=_integer(topk,'topk')
    detector,classifier=DetectorNet(),CaseNet(topk)
    restore_checkpoint(detector,detector_checkpoint)
    restore_checkpoint(classifier,classifier_checkpoint)
    detector.eval();classifier.eval()
    return dict(detector=detector,classifier=classifier)
