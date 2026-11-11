"""Qualify backbone state, feature maps and gradients against historical source."""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path

import torch
from torchvision.models._utils import IntermediateLayerGetter

from sciona.wheat_resnet_backbone import initialized_resnet152, CHECKPOINT_SHA256


def source_classes(path, digest, names, namespace):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('historical source drift')
    nodes = [n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('historical source definitions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-backbone-source>', 'exec'), namespace)


def main(source_root, checkpoint, output):
    torch.set_num_threads(2)
    namespace = dict(torch=torch, nn=torch.nn)
    source_classes(source_root / 'ops/misc.py',
        'f75578efe5c4054e2fdd479db15bfb5cca15b610b909185b40f452c4b87a91db', ['FrozenBatchNorm2d'], namespace)
    source_classes(source_root / 'models/resnet.py',
        '4e1b72ec251835ac5f6a6c3f32a378fa9168efc9dd20953af42ed140a7344a37',
        ['conv1x1', 'conv3x3', 'BasicBlock', 'Bottleneck', 'ResNet'], namespace)
    actual = initialized_resnet152(checkpoint)
    state = actual.state_dict()
    reference = namespace['ResNet'](namespace['Bottleneck'], [3, 8, 36, 3], norm_layer=namespace['FrozenBatchNorm2d'])
    reference.load_state_dict(state, strict=True)
    for name, parameter in reference.named_parameters():
        parameter.requires_grad_(any(stage in name for stage in ('layer2', 'layer3', 'layer4')))
    assert {n: p.requires_grad for n, p in actual.named_parameters()} == {n: p.requires_grad for n, p in reference.named_parameters()}
    tensor_count = len(state)
    for key, value in state.items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    layers = {f'layer{i}': str(i) for i in range(1, 5)}
    actual = IntermediateLayerGetter(actual, return_layers=layers)
    reference = IntermediateLayerGetter(reference, return_layers=layers)
    # The source detector internally resizes square 1024 inputs to 800.
    x = torch.rand(1, 3, 800, 800, generator=torch.Generator().manual_seed(492))
    with torch.no_grad():
        expected = reference(x)
    del reference, state
    gc.collect()
    observed = actual(x)
    shapes = {}
    for key in layers.values():
        torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0)
        assert torch.isfinite(observed[key]).all()
        shapes[key] = list(observed[key].shape)
    observed['4'].square().mean().backward()
    trainable = [p for p in actual.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert all(p.grad is None for p in actual.parameters() if not p.requires_grad)
    assert any(p.grad.abs().sum() > 0 for p in trainable)
    files = ['sciona/wheat_legacy_weights.py', 'sciona/wheat_resnet_backbone.py', 'sciona/wheat_frozen_normalization.py', 'scripts/validate_wheat_resnet_backbone.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        checkpoint_sha256=CHECKPOINT_SHA256, initialized_tensor_count=tensor_count,
        historical_feature_maps_exact=shapes, freezing_matches_source=True,
        full_trainable_backbone_gradients_finite=True, trainable_parameter_tensors=len(trainable),
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Backbone only; FPN, RPN, ROI heads, detector training and inference remain unqualified.',
                'Historical Python architecture runs on installed Torch; historical native-kernel equivalence is not claimed.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.checkpoint, args.output)
