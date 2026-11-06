"""Complete unchanged four-family training, with final arithmetic exposed as graph dataflow."""
import gc
from pathlib import Path

from sciona.cassava_fold_contract import _keys
from sciona.cassava_population_views import align_training_views, _align


def train_families(plan, *, source_rows, efficientnet_rows, query_rows, references,
            torch_batch_size, torch_workers, efficientnet_batch_size, output_directory):
    """Run all CV families, final B4 refit and frozen CropNet before assembly.

Runtime inputs and output checkpoints remain private. The caller must establish
preparation provenance and supply reviewed artifact hashes. This CPU execution
retains the documented corrections to the source population lifecycle.
"""
    views = align_training_views(plan, source_rows=source_rows, efficientnet_rows=efficientnet_rows)
    if not isinstance(query_rows, (list, tuple)) or not query_rows:
        raise ValueError('Nonempty keyed query image rows required')
    if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in query_rows):
        raise ValueError('Each query row must contain identity and encoded image')
    keys = _keys([row[0] for row in query_rows])
    queries = _align(keys, query_rows, (600, 800, 3))
    if not isinstance(references, dict) or set(references) != {'vit', 'resnext', 'efficientnet', 'cropnet'}:
        raise ValueError('Reviewed references for all four model families required')
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    from sciona.cassava_torch_folds import train_five_folds as torch_folds
    outputs, artifacts, histories = {}, {}, {}
    for family in ('resnext', 'vit'):
        ref = references[family]
        mean, _, history, checkpoints = torch_folds(family, plan, views.source_images, keys, queries,
            weights_path=ref['path'], expected_sha256=ref['sha256'], batch_size=torch_batch_size,
            workers=torch_workers, output_directory=directory / family)
        outputs[family] = (keys, mean)
        histories[family], artifacts[family] = history, checkpoints
        gc.collect()
    from sciona.cassava_efficientnet_folds import train_five_folds as efficientnet_folds
    ref = references['efficientnet']
    history, evaluations, checkpoints = efficientnet_folds(plan, views.efficientnet_images,
        weights=ref['path'], expected_sha256=ref['sha256'], batch_size=efficientnet_batch_size,
        output_directory=directory / 'efficientnet-cv')
    histories['efficientnet_cv'], artifacts['efficientnet_cv'] = history, checkpoints
    from sciona.cassava_efficientnet_refit import train_final, predict_images
    from sciona.cassava_resnext_images import decode_rgb
    model, history, checkpoint = train_final(views.efficientnet_images, plan.labels,
        weights=ref['path'], expected_sha256=ref['sha256'], weight_format='upstream',
        batch_size=efficientnet_batch_size, output_directory=directory / 'efficientnet-final')
    decoded = [decode_rgb(image) for image in queries]
    outputs['efficientnet'] = (keys, predict_images(model, decoded))
    histories['efficientnet_final'], artifacts['efficientnet_final'] = history, checkpoint
    del model
    gc.collect()
    from sciona.cassava_cropnet import load_model, predict_images as cropnet_predict
    ref = references['cropnet']
    model = load_model(ref['directory'], expected_hashes=ref['files'])
    outputs['cropnet'] = (keys, cropnet_predict(model, decoded))
    return dict(query_keys=keys, family_outputs=outputs, histories=histories,
                evaluations=evaluations, checkpoints=artifacts)
