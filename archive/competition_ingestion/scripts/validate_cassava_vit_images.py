"""Source configuration and pixel comparisons using synthetic RGB arrays."""
import ast
import hashlib
import json
import random
import textwrap
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

from sciona.cassava_vit_images import transformations
from scripts.validate_cassava_source_components import source_helpers


def main():
    cache = Path('/private/tmp/sciona_cassava_winner_source')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    for name in ('vit', 'inference'):
        assert hashlib.sha256((cache / (name + '.py')).read_bytes()).hexdigest() == pins['notebooks'][name]['code_sha256']
    names = ['Compose', 'RandomResizedCrop', 'Transpose', 'HorizontalFlip', 'VerticalFlip', 'ShiftScaleRotate',
             'HueSaturationValue', 'RandomBrightnessContrast', 'Normalize', 'CoarseDropout', 'Cutout', 'CenterCrop', 'Resize']
    namespace = {name: getattr(A, name) for name in names}
    namespace.update(ToTensorV2=ToTensorV2, CFG={'img_size': 384}, vit_image_size=384)
    ns = source_helpers(cache / 'vit.py', {'get_train_transforms', 'get_valid_transforms'}, namespace)
    source = (cache / 'inference.py').read_text()
    start = source.index('    def get_tta_transforms():')
    end = source.index('    def inference(', start)
    tree = ast.parse(textwrap.dedent(source[start:end]))
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    exec(compile(tree, '<pinned-v it-inference-transform>', 'exec'), ns)
    raw = np.random.default_rng(733).integers(0, 256, (600, 800, 3), dtype=np.uint8)
    validation_views = []
    for stage, source_fn in [('train', 'get_train_transforms'), ('validation', 'get_valid_transforms'), ('inference', 'get_tta_transforms')]:
        candidate, reference = transformations(stage), ns[source_fn]()
        assert candidate.to_dict() == reference.to_dict()
        for seed in range(20 if stage != 'inference' else 1):
            random.seed(seed); np.random.seed(seed)
            actual = candidate(image=raw)['image']
            random.seed(seed); np.random.seed(seed)
            expected = reference(image=raw)['image']
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.shape == (3, 384, 384) and torch.isfinite(actual).all()
            if stage == 'validation': validation_views.append(actual)
    assert any(not torch.equal(validation_views[0], x) for x in validation_views[1:])
    reference_pixels = raw[108:492, 208:592].astype(np.float64)
    oracle = ((reference_pixels / 255. - [.485, .456, .406]) / [.229, .224, .225]).transpose(2, 0, 1)
    np.testing.assert_allclose(transformations('inference')(image=raw)['image'].numpy(), oracle, atol=5e-7)
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
              'source_transform_comparisons': 41, 'configuration_match': True,
              'validation_stochasticity_verified': True, 'inference_crop_scalar_oracle': True,
              'scope': 'Reference transform parity; historical augmentation version and complete ViT fold remain unqualified.'}
    paths = ['sciona/cassava_vit_images.py', 'scripts/validate_cassava_vit_images.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_vit_images_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
