"""Actual full pretrained four-family workflow, synthetic inputs only."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import tempfile

import cv2
import numpy as np
import tensorflow as tf
import torch

from sciona.cassava_pipeline import execute
from sciona.cassava_fold_contract import build_plan
from sciona.cassava_torch_folds import predict_images
from sciona.cassava_resnext import Classifier as ResNeXt
from sciona.cassava_vit_model import Classifier as ViT


def main():
    torch.set_num_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    torch.manual_seed(718)
    random.seed(718)
    np.random.seed(718)
    tf.keras.utils.set_random_seed(819)
    timm_pin = json.loads(Path('docs/reviews/competition_cassava_timm_pretrained_validation.json').read_text())
    efficientnet_pin = json.loads(Path('/private/tmp/sciona_cassava_efficientnet_upstream_reference/manifest.json').read_text())
    cropnet_pin = json.loads(Path('docs/reviews/competition_cassava_cropnet_artifact.json').read_text())
    references = {}
    for family, tag in [('resnext', 'resnext50_32x4d.ra_in1k'), ('vit', 'vit_base_patch16_384.orig_in21k_ft_in1k')]:
        references[family] = {'path': Path('/private/tmp/sciona_cassava_timm_pretrained') / f'{tag}.safetensors',
                              'sha256': timm_pin['references'][tag]['weights_sha256']}
    references['efficientnet'] = {'path': Path('/private/tmp/sciona_cassava_efficientnet_upstream_reference/model.h5'),
                                  'sha256': efficientnet_pin['sha256']}
    references['cropnet'] = {'directory': Path('/private/tmp/sciona_cassava_cropnet_reference/model'), 'files': cropnet_pin['files']}
    def jpeg(value, shape):
        ok, encoded = cv2.imencode('.jpg', np.full(shape, value, np.uint8))
        assert ok
        return encoded.tobytes()
    keys = [f'synthetic-{i}' for i in range(25)]
    plan = build_plan(keys, [i % 5 for i in range(25)], [i // 5 for i in range(25)])
    # Constant synthetic canvases have an explicit common generating value;
    # no assertion about the historical prepared-record recipe is made.
    source = [(key, jpeg(70 + i, (600, 800, 3))) for i, key in enumerate(keys)]
    prepared = [(key, jpeg(70 + i, (512, 512, 3))) for i, key in enumerate(keys)]
    queries = [(f'synthetic-query-{i}', jpeg(110 + i * 20, (600, 800, 3))) for i in range(3)]
    with tempfile.TemporaryDirectory(prefix='cassava-complete-pipeline-') as directory:
        print('START complete workflow: 15 CV fits, 14-epoch final refit and frozen CropNet', flush=True)
        result, outputs, histories, evaluations, artifacts = execute(plan,
            source_rows=source[::-1], efficientnet_rows=prepared[7:] + prepared[:7], query_rows=queries,
            references=references, torch_batch_size=1, torch_workers=0, efficientnet_batch_size=1,
            output_directory=Path(directory) / 'execution')
        print('TRAINING COMPLETE; verifying checkpoint and ensemble results', flush=True)
        assert set(outputs) == {'vit', 'resnext', 'efficientnet', 'cropnet'}
        for family, epochs, classifier in [('resnext', 15, ResNeXt), ('vit', 10, ViT)]:
            replayed = []
            assert set(histories[family]) == set(range(5))
            for fold in range(5):
                history = histories[family][fold]
                assert len(history) == epochs
                scores = [record['accuracy'] for record in history]
                best = scores.index(max(scores)) if family == 'resnext' else max(i for i, value in enumerate(scores) if value == max(scores))
                saved = torch.load(artifacts[family][fold], map_location='cpu', weights_only=True)
                assert saved['epoch'] == best
                model = classifier(pretrained=False)
                model.load_state_dict(saved['model'], strict=True)
                replayed.append(predict_images(family, model, [row[1] for row in queries], batch_size=1))
                del model, saved
                gc.collect()
            np.testing.assert_array_equal(np.stack(replayed).mean(0), outputs[family][1])
        assert len(histories['efficientnet_final']['loss']) == 14
        assert set(histories['efficientnet_cv']) == set(range(5))
        cv_epochs = []
        for fold in range(5):
            history = histories['efficientnet_cv'][fold]
            assert 1 <= len(history['loss']) <= 20
            best = int(np.argmax(history['val_categorical_accuracy']))
            np.testing.assert_allclose(evaluations[fold]['categorical_accuracy'], history['val_categorical_accuracy'][best], atol=1e-7)
            cv_epochs.append(len(history['loss']))
        expected = np.array([[(float(outputs['vit'][1][r,c]) + float(outputs['resnext'][1][r,c])) / 2
                              + float(outputs['efficientnet'][1][r,c]) + float(outputs['cropnet'][1][r,c])
                              for c in range(5)] for r in range(3)])
        np.testing.assert_array_equal(result['scores'], expected)
        np.testing.assert_array_equal(result['labels'], expected.argmax(-1))
        np.testing.assert_allclose(result['scores'].sum(-1), 3., atol=1e-6)
        count = sum(len(artifacts[key]) for key in ('resnext', 'vit', 'efficientnet_cv')) + 1
        assert count == 16 and artifacts['efficientnet_final'].is_file()
    # Snapshot all Cassava runtime modules so changes to any transitive branch
    # require deliberate reevaluation of this full-workflow evidence.
    paths = sorted(str(p) for p in Path('sciona').glob('cassava_*.py'))
    paths += ['scripts/validate_cassava_pipeline.py', 'docs/reviews/competition_cassava_timm_pretrained_validation.json',
              'docs/reviews/competition_cassava_cropnet_artifact.json']
    report = {'approved': False, 'passed': True, 'synthetic_only': True,
        'trainable_model_fits': 16, 'frozen_models': 1, 'torch_folds_per_family': 5,
        'resnext_epochs_per_fold': 15, 'vit_epochs_per_fold': 10, 'efficientnet_cv_epochs': cv_epochs,
        'efficientnet_refit_epochs': 14, 'prediction_images': 3,
        'torch_checkpoint_prediction_replays_exact': True, 'final_scalar_assembly_exact': True,
        'temporary_checkpoints_removed': True,
        'runtime': {name: importlib.metadata.version(name) for name in ['torch', 'timm', 'tensorflow', 'tf-keras', 'numpy', 'safetensors']},
        'scope': 'Complete pretrained CPU reconstruction with diagnostic batch one, zero torch workers and synthetic paired image representations. Historical preprocessing provenance, original runtime/budget parity, dependency/license and graph publication gates remain separate.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_pipeline_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS complete pretrained four-family workflow', flush=True)


if __name__ == '__main__':
    main()
