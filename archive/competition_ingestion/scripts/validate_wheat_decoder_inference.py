"""Check restored integer indices and full D6 TTA-to-fusion inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from sciona.wheat_legacy_decode import legacy_postprocess
from sciona.wheat_tta import WheatTTA
from sciona.wheat_prediction_boundary import prepare_prediction
from sciona.wheat_fusion import fuse_predictions


def main(source_root, dependency_root, output):
    prior = json.loads(Path('docs/reviews/competition_global_wheat_detector_execution.json').read_text())
    if prior.get('passed') is not True:
        raise ValueError('detector execution prerequisite missing')
    for name, digest in prior['runtime_source_sha256'].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('runtime source differs from checked execution')
    for package, manifest in prior['publisher_dependencies'].items():
        for name, digest in manifest['files_sha256'].items():
            if hashlib.sha256((dependency_root / package / name).read_bytes()).hexdigest() != digest:
                raise ValueError('dependency payload differs')
    helper = dependency_root / 'torch14_six.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest() != prior['historical_torch_six_sha256']:
        raise ValueError('compatibility helper differs')
    spec = importlib.util.spec_from_file_location('torch._six', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['torch._six'] = module
    from effdet import bench
    import models
    if Path(bench.__file__).resolve() != (source_root / 'effdet/bench.py').resolve():
        raise ValueError('unexpected detector implementation')
    restored = legacy_postprocess(source_root / 'effdet/bench.py')
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(132)
    index_cases = 0
    for classes in [1, 3]:
        config = SimpleNamespace(num_classes=classes, num_levels=5)
        cls = [torch.rand(2, 9 * classes, s, s, generator=generator, dtype=torch.float64) for s in [32, 16, 8, 4, 2]]
        box = [torch.rand(2, 36, s, s, generator=generator, dtype=torch.float64) for s in [32, 16, 8, 4, 2]]
        selected_cls, selected_box, indices, labels = restored(config, cls, box)
        all_cls = np.concatenate([x.numpy().transpose(0, 2, 3, 1).reshape(2, -1, classes) for x in cls], axis=1)
        all_box = np.concatenate([x.numpy().transpose(0, 2, 3, 1).reshape(2, -1, 4) for x in box], axis=1)
        for batch in range(2):
            flat = np.argsort(all_cls[batch].reshape(-1))[::-1][:5000]
            anchors, category = np.divmod(flat, classes)
            np.testing.assert_array_equal(indices[batch], anchors)
            np.testing.assert_array_equal(labels[batch], category)
            np.testing.assert_array_equal(selected_cls[batch, :, 0], all_cls[batch, anchors, category])
            np.testing.assert_array_equal(selected_box[batch], all_box[batch, anchors])
            index_cases += 1
    bench._post_process = restored
    torch.manual_seed(133)
    model = models.get_effdet_test('ed6', img_size=640).eval()
    image = torch.rand(1, 3, 640, 640, generator=generator)
    boxes, scores, counts = [], [], []
    with torch.no_grad():
        for view in range(8):
            prediction = model(WheatTTA(view=view, image_size=640).augment_tensor(image), torch.ones(1))
            if prediction.shape != (1, 100, 6) or not torch.isfinite(prediction).all():
                raise ValueError('detector inference output contract differs')
            raw = prediction[0].numpy()
            b, s, _ = prepare_prediction(raw[:, :4], raw[:, 4], detector='effdet', view=view, image_size=640)
            boxes.append(b); scores.append(s); counts.append(len(b))
    fused_boxes, fused_scores = fuse_predictions(boxes, scores, stage='pseudo1', height=640, width=640)
    if not np.isfinite(fused_boxes).all() or not np.isfinite(fused_scores).all():
        raise ValueError('nonfinite fused output')
    files = ['sciona/wheat_legacy_decode.py', 'sciona/wheat_tta.py', 'sciona/wheat_prediction_boundary.py',
             'sciona/wheat_fusion.py', 'scripts/validate_wheat_decoder_inference.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        pretrained_weights=False, independent_index_cases=index_cases, indices_checked=index_cases * 5000,
        index_and_selected_tensor_values_exact=True, full_d6_inference_views=8,
        detections_per_view=counts, fused_detections=len(fused_boxes),
        compatibility_change='Single pinned index division uses floor division, preserving historical nonnegative integer semantics.',
        semantics_reference='https://github.com/pytorch/pytorch/releases/tag/v1.6.0',
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Random D6 inference through eight TTA views, detector boundary and fusion; no recognition or trained-model claim.',
                'Historical NMS/kernel equivalence and remaining detector configurations require separate qualification.',
                'Full training and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'implementation_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dependency-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.dependency_root, args.output)
