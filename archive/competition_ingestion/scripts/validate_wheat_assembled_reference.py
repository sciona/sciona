"""Compare assembled detector with pinned historical Python architecture."""
import argparse
import ast
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Dict, Tuple
import warnings

import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops, roi_align

from sciona.wheat_fasterrcnn import initialized_fasterrcnn
from sciona.wheat_legacy_weights import read_legacy_resnet152
from sciona.wheat_resnet_backbone import CHECKPOINT_SHA256

SOURCE_HASHES = {'models/detection/backbone_utils.py': '698089e302ee1c07b3392bddf06d1b96a51c864679f3f44d9d3a1338e97dc2e5', 'models/detection/faster_rcnn.py': 'b9df7856bad4aa091e67d04d363ea85e2bbf47ca4d0008f58418d6b8d304ff59', 'models/detection/rpn.py': '62ea420ac8bba44d91d3b27f46c627fcc6b91c15ae706142fcbe079c10a31751', 'models/detection/roi_heads.py': 'dfe9ac02abbbd2b6db522d66d00210ad0d7dac1cc66cf1f0c3e9bd628f37be2c', 'models/detection/transform.py': '46024143aa8d85210135e659fc6487b642cd5d8b0296cc4b7f29f1a329a836aa', 'models/resnet.py': '4e1b72ec251835ac5f6a6c3f32a378fa9168efc9dd20953af42ed140a7344a37', 'ops/feature_pyramid_network.py': 'd5acedbbc2b192516044c9d655e1757dfecea662c518b6de9dfc5056f50f8d2e', 'ops/misc.py': 'f75578efe5c4054e2fdd479db15bfb5cca15b610b909185b40f452c4b87a91db', 'models/detection/_utils.py': 'cb80cd35f7f0ba41e5929c1bf37b1096ce6d2f0d1a34954d9dfa8d9ac3e29484', 'ops/poolers.py': 'f3f65f56bf3a3e2fe9b0d5e9082b779fd8344e7ed4b5fb2a8fcb5215ca5af1aa', 'ops/roi_align.py': 'c3e92d1db2f418d17bb7f20887d99e4644b44df1e7cd7bd6b5021f7751acdc15', 'models/detection/generalized_rcnn.py': 'c8d3c0bfa7feb914e2fe5d72115c3667941b396a513de7552352b97ff8559380', 'torch14_functional.py': '45d25e49078f0c00bd559d3d7ff412d9250d6f4ad1faf406bb4abab968db4cfb'}


def historical_model(root, checkpoint):
    common = dict(torch=torch, torchvision=torchvision, nn=torch.nn, Tensor=torch.Tensor,
        F=torch.nn.functional, math=math, warnings=warnings, OrderedDict=OrderedDict,
        Optional=Optional, List=List, Dict=Dict, Tuple=Tuple, ImageList=ImageList,
        box_ops=box_ops, box_area=box_ops.box_area, roi_align=roi_align,
        IntermediateLayerGetter=IntermediateLayerGetter)
    def load(name, names=None, **extra):
        raw = (root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError('historical architecture source drift: '+name)
        nodes = [n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and (names is None or n.name in names)]
        for node in nodes:
            node.decorator_list = []
        namespace = dict(common, **extra)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical:'+name+'>', 'exec'), namespace)
        return namespace
    bn = load('ops/misc.py', {'FrozenBatchNorm2d'})
    resnet = load('models/resnet.py', {'conv1x1', 'conv3x3', 'BasicBlock', 'Bottleneck', 'ResNet'})
    fpn = load('ops/feature_pyramid_network.py')
    backbone_utils = load('models/detection/backbone_utils.py', {'BackboneWithFPN'},
        FeaturePyramidNetwork=fpn['FeaturePyramidNetwork'], LastLevelMaxPool=fpn['LastLevelMaxPool'])
    utils = load('models/detection/_utils.py')
    rpn = load('models/detection/rpn.py', det_utils=SimpleNamespace(**utils))
    roi = load('models/detection/roi_heads.py', det_utils=SimpleNamespace(**utils))
    pool = load('ops/poolers.py')
    old_functional = load('torch14_functional.py', {'interpolate'}, __package__='torch.nn')
    class TorchProxy:
        nn = SimpleNamespace(functional=SimpleNamespace(interpolate=old_functional['interpolate']))
        def __getattr__(self, key):
            return getattr(torch, key)
    transform = load('models/detection/transform.py', {'GeneralizedRCNNTransform', 'resize_boxes'}, torch=TorchProxy())
    generalized = load('models/detection/generalized_rcnn.py')
    detector = load('models/detection/faster_rcnn.py', {'FasterRCNN', 'TwoMLPHead', 'FastRCNNPredictor'},
        GeneralizedRCNN=generalized['GeneralizedRCNN'], AnchorGenerator=rpn['AnchorGenerator'],
        RPNHead=rpn['RPNHead'], RegionProposalNetwork=rpn['RegionProposalNetwork'], RoIHeads=roi['RoIHeads'],
        MultiScaleRoIAlign=pool['MultiScaleRoIAlign'], GeneralizedRCNNTransform=transform['GeneralizedRCNNTransform'])
    raw = checkpoint.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('pretrained artifact differs')
    body = resnet['ResNet'](resnet['Bottleneck'], [3, 8, 36, 3], norm_layer=bn['FrozenBatchNorm2d'])
    body.load_state_dict(read_legacy_resnet152(raw), strict=True)
    for name, parameter in body.named_parameters():
        parameter.requires_grad_(any(stage in name for stage in ('layer2', 'layer3', 'layer4')))
    backbone = backbone_utils['BackboneWithFPN'](body, {f'layer{i}':str(i-1) for i in range(1,5)}, [256,512,1024,2048], 256)
    model = detector['FasterRCNN'](backbone, 91)
    model.roi_heads.box_predictor = detector['FastRCNNPredictor'](1024, 2)
    return model


def main(root, checkpoint, output):
    torch.set_num_threads(2)
    torch.manual_seed(1287)
    actual = initialized_fasterrcnn(checkpoint)
    torch.manual_seed(1287)
    reference = historical_model(root, checkpoint)
    def canonical(key):
        return key.replace('rpn.head.conv.0.0.', 'rpn.head.conv.')
    states = {canonical(k):v for k,v in actual.state_dict().items()}
    assert set(states) == set(reference.state_dict())
    for key, value in states.items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    assert {canonical(k):p.requires_grad for k,p in actual.named_parameters()} == {k:p.requires_grad for k,p in reference.named_parameters()}
    print(json.dumps(dict(complete_initialization_exact=True, tensors=len(states))), flush=True)
    images = [torch.rand(3,1024,1024,generator=torch.Generator().manual_seed(1288))]
    targets = [dict(boxes=torch.tensor([[80.,90.,280.,310.],[430.,470.,810.,890.]]),labels=torch.ones(2,dtype=torch.int64))]
    actual.train(); reference.train()
    torch.manual_seed(1290)
    observed = actual(images, copy.deepcopy(targets))
    observed_rng = torch.get_rng_state()
    sum(observed.values()).backward()
    torch.manual_seed(1290)
    expected = reference(images, copy.deepcopy(targets))
    torch.testing.assert_close(observed_rng, torch.get_rng_state(), rtol=0, atol=0)
    assert set(observed) == set(expected)
    for key in observed:
        torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0)
    sum(expected.values()).backward()
    reference_parameters = dict(reference.named_parameters())
    gradients = 0
    for key, parameter in actual.named_parameters():
        other = reference_parameters[canonical(key)]
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
            assert torch.isfinite(parameter.grad).all()
            gradients += 1
        else:
            assert parameter.grad is None and other.grad is None
    print(json.dumps(dict(losses_and_parameter_gradients_exact=True, gradients=gradients)), flush=True)
    actual.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
    actual.eval(); reference.eval()
    with torch.no_grad():
        a, b = actual(images), reference(images)
    for x,y in zip(a,b):
        assert set(x)==set(y)
        for key in x:
            torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        input_size=1024, initialized_tensors_exact=len(states), losses_exact=list(observed),
        parameter_gradients_exact=gradients, training_rng_exact=True, inference_exact=True,
        source_sha256=SOURCE_HASHES, validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['One synthetic full-resolution batch; complete source training budgets and ensemble remain pending.',
                'Historical Python architecture uses installed native kernels and eager execution; historical native-kernel/TorchScript parity is not claimed.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.source_root,args.checkpoint,args.output)
