"""Execute all five pretrained folds for both torch ensemble families."""
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import cv2
import numpy as np
import torch

from sciona.cassava_fold_contract import build_plan, members
from sciona.cassava_torch_folds import train_five_folds, predict_images
from sciona.cassava_resnext import Classifier as ResNeXt
from sciona.cassava_vit_model import Classifier as ViT


def main():
    torch.set_num_threads(4)
    torch.manual_seed(718)
    random.seed(718)
    np.random.seed(718)
    artifact_path = Path('docs/reviews/competition_cassava_timm_pretrained_validation.json')
    artifact = json.loads(artifact_path.read_text())
    assert artifact['passed'] and not artifact['approved']
    for path, expected in artifact['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    def jpeg(value):
        ok, encoded = cv2.imencode('.jpg', np.full((600, 800, 3), value, np.uint8))
        assert ok
        return encoded.tobytes()
    # Distinct synthetic image contents, with every class in every held-out fold.
    images = [jpeg(70 + i) for i in range(25)]
    assert len(set(images)) == 25
    plan = build_plan([f'synthetic-{i}' for i in range(25)],
                      [i % 5 for i in range(25)], [i // 5 for i in range(25)])
    prediction_images = [jpeg(110), jpeg(130), jpeg(150)]
    keys = ['synthetic-query-a', 'synthetic-query-b', 'synthetic-query-c']
    results = {}
    with tempfile.TemporaryDirectory(prefix='cassava-torch-fivefold-') as temp:
        for family, classifier, epochs, tag in [
            ('resnext', ResNeXt, 15, 'resnext50_32x4d.ra_in1k'),
            ('vit', ViT, 10, 'vit_base_patch16_384.orig_in21k_ft_in1k'),
        ]:
            print(f'START {family}: five complete pretrained folds', flush=True)
            reference = artifact['references'][tag]
            combined, folds, histories, checkpoints = train_five_folds(family, plan, images, keys,
                prediction_images, weights_path=Path('/private/tmp/sciona_cassava_timm_pretrained') / f'{tag}.safetensors',
                expected_sha256=reference['weights_sha256'], batch_size=1, workers=0,
                output_directory=Path(temp) / family)
            best_epochs = []
            for fold in range(5):
                train, valid = members(plan, fold)
                assert not set(train) & set(valid) and len(train) == 20 and len(valid) == 5
                history = histories[fold]
                assert len(history) == epochs
                assert all(math.isfinite(value) for row in history for value in row.values())
                scores = [row['accuracy'] for row in history]
                best = (scores.index(max(scores)) if family == 'resnext'
                        else max(i for i, value in enumerate(scores) if value == max(scores)))
                saved = torch.load(checkpoints[fold], map_location='cpu', weights_only=True)
                assert saved['epoch'] == best
                model = classifier(pretrained=False)
                model.load_state_dict(saved['model'], strict=True)
                replay = predict_images(family, model, prediction_images, batch_size=1)
                np.testing.assert_array_equal(replay, folds[fold][1])
                best_epochs.append(best)
                del model, saved
                gc.collect()
            # Independent scalar high-precision mean, not another call to the reducer.
            expected = np.array([[math.fsum(float(folds[f][1][row, col]) for f in range(5)) / 5
                                  for col in range(5)] for row in range(3)])
            np.testing.assert_allclose(combined, expected, atol=1e-7, rtol=1e-6)
            results[family] = {'folds_executed': 5, 'epochs_per_fold': epochs,
                'training_images_per_fold': 20, 'validation_images_per_fold': 5,
                'prediction_images': 3, 'best_epochs': best_epochs,
                'all_checkpoint_predictions_replayed_exactly': True,
                'fivefold_mean_matches_scalar_oracle': True,
                'pretrained_sha256': reference['weights_sha256']}
            print(f'PASS {family}: all five checkpoints and probability means verified', flush=True)
    paths = ['sciona/cassava_torch_folds.py', 'sciona/cassava_fold_contract.py',
        'sciona/cassava_fold_dispatch.py', 'sciona/cassava_pretrained.py',
        'sciona/cassava_resnext.py', 'sciona/cassava_resnext_fold.py', 'sciona/cassava_resnext_images.py',
        'sciona/cassava_vit_model.py', 'sciona/cassava_vit_fold.py', 'sciona/cassava_vit_images.py',
        'sciona/cassava_vit_ordered_loss.py', 'sciona/cassava_loss.py',
        'scripts/validate_cassava_torch_folds.py', str(artifact_path)]
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
        'pretrained_weights_used': True, 'diagnostic_batch_size': 1, 'workers': 0,
        'families': results,
        'scope': 'Complete pretrained fivefold CPU reconstruction for two families with synthetic populations. Original budgets, TPU parity, physical identity ingestion, EfficientNet and full four-family ensemble remain unqualified.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_torch_folds_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS both pretrained fivefold families; temporary checkpoints removed', flush=True)


if __name__ == '__main__':
    main()
