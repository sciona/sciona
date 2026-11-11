"""Compare safe initialization with the pinned base-training factory."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from sciona.wheat_initialization import initialized_detector, CHECKPOINTS


def main(source_root, dependency_root, checkpoint_root, output):
    prior = json.loads(Path('docs/reviews/competition_global_wheat_detector_execution.json').read_text())
    for name, digest in prior['runtime_source_sha256'].items():
        if hashlib.sha256((source_root / 'runtime' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('runtime source drift')
    for package, manifest in prior['publisher_dependencies'].items():
        for name, digest in manifest['files_sha256'].items():
            if hashlib.sha256((dependency_root / package / name).read_bytes()).hexdigest() != digest:
                raise ValueError('dependency source drift')
    helper = dependency_root / 'torch14_six.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest() != prior['historical_torch_six_sha256']:
        raise ValueError('compatibility source drift')
    spec = importlib.util.spec_from_file_location('torch._six', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['torch._six'] = module
    from effdet import EfficientDet, get_efficientdet_config, DetBenchTrain, DetBenchEval
    from effdet.efficientdet import HeadNet
    raw = (source_root / 'models.py').read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'c9373461d6fc144ffe277d9975444f42c1a346d08dd8f9077988eac7fde3812b':
        raise ValueError('base-training factory differs')
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'get_effdet')
    code = compile(ast.Module(body=[node], type_ignores=[]), '<pinned-base-factory>', 'exec')
    receipts = json.loads((checkpoint_root / 'manifest.json').read_text())
    torch.set_num_threads(2)
    results = []
    for backbone, size in [('ed5', 512), ('ed6', 640), ('ed7', 768)]:
        receipt = next(r for r in receipts if r['sha256'] == CHECKPOINTS[backbone])
        path = checkpoint_root / receipt['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
            raise ValueError('pretrained source artifact drift')
        state = torch.load(path, map_location='cpu', weights_only=True)
        def reference_load(requested):
            if Path(requested).name != receipt['file']:
                raise ValueError('factory requested another checkpoint')
            return state
        namespace = dict(torch=SimpleNamespace(load=reference_load), gc=gc,
            EfficientDet=EfficientDet, get_efficientdet_config=get_efficientdet_config,
            DetBenchTrain=DetBenchTrain, DetBenchEval=DetBenchEval, HeadNet=HeadNet)
        exec(code, namespace)
        torch.manual_seed(271)
        actual = initialized_detector(backbone, path, image_size=size).eval()
        torch.manual_seed(271)
        reference = namespace['get_effdet'](backbone, img_size=size).eval()
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
        retained = 0
        for key, value in actual.model.state_dict().items():
            if not key.startswith('class_net.'):
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                retained += 1
        del reference, state, namespace
        images = torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(272)).requires_grad_()
        loss, classification, box = actual(images, [torch.tensor([[50., 60., 200., 220.]])], [torch.tensor([1.])])
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actual.parameters() if p.requires_grad)
        result = dict(backbone=backbone, size=size, checkpoint_sha256=receipt['sha256'],
            complete_source_factory_state_exact=True, retained_pretrained_tensors_exact=retained,
            fresh_one_class_tower_matches_source=True, full_loss_and_backward_passed=True)
        results.append(result)
        print(json.dumps(result), flush=True)
        del actual, images, loss, classification, box
        gc.collect()
    files = ['sciona/wheat_initialization.py', 'scripts/validate_wheat_initialization.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        configurations=results, source_factory_sha256=hashlib.sha256(raw).hexdigest(),
        publisher_artifacts=receipts,
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Source-prescribed pretrained detector initialization only; new classification towers remain untrained.',
                'Both factory paths use the qualified private source overlay on modern Torch; historical kernel equivalence remains separate.',
                'Artifact licensing, full detector training, ensemble integration and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'configurations': len(results), 'approved': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dependency-root', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.dependency_root, args.checkpoint_root, args.output)
