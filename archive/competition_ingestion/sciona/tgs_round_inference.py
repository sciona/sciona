"""Connect reviewed TGS phase receipts to complete pre-mosaic ensembles."""
import numpy as np

from sciona.tgs_checkpoint_selection import ensemble_selection, average_torch_folds
from sciona.tgs_ensemble import blend_predictions
from sciona.tgs_keras_inference import predict_checkpoints
from sciona.tgs_torch_inference import predict_snapshots


def round_receipts(stage, completed_fits):
    """Require exact cycle boundaries, including unselected cycles.

    Accept JSON-round-tripped epoch keys without silently shifting a missing
    cycle into a different zero-based snapshot index.
    """
    selection = ensemble_selection(stage)
    keras = []
    for group in selection['keras']:
        checkpoints = []
        for item in group:
            fit = completed_fits.get(item['fit'], {})
            if 'best_checkpoint' not in fit:
                raise ValueError('missing validation-best checkpoint: ' + item['fit'])
            checkpoints.append(fit['best_checkpoint'])
        keras.append(checkpoints)
    torch = []
    expected = list(range(50, 301 if stage == 1 else 201, 50))
    for item in selection['pytorch']:
        fit = completed_fits.get(item['fit'], {})
        raw = fit.get('cycle_checkpoints', {})
        if not isinstance(raw, dict) or any(str(key) not in {str(n) for n in expected} for key in raw):
            raise ValueError('invalid source cycle boundaries')
        cycles = {int(key): value for key, value in raw.items()}
        if sorted(cycles) != expected or len(cycles) != len(raw):
            raise ValueError('missing or duplicated source cycle boundary: ' + item['fit'])
        torch.append([cycles[expected[index]] for index in item['indices']])
    return dict(keras=keras, pytorch=torch)


def predict_round(stage, completed_fits, store, keras_model, torch_model,
                  keras_images, torch_images, constant_images, *, keras_batch_size, torch_batch_size):
    """Execute every selected checkpoint, TTA, fold mean and branch blend.

    Inputs are the same ordered population in the two branch representations.
    Stage3 output still requires the reviewed mosaic postprocessing operation.
    Models remain in eval mode with the final selected checkpoint loaded.
    """
    receipts = round_receipts(stage, completed_fits)
    if torch_model.variant != {1: 4, 2: 3, 3: 5}[stage]:
        raise ValueError('incorrect PyTorch architecture for ensemble round')
    if len(keras_images) != len(torch_images) or len(constant_images) != len(torch_images):
        raise ValueError('aligned branch populations required')
    keras_predictions = []
    for group in receipts['keras']:
        folds = []
        for receipt in group:
            folds.append(predict_checkpoints(keras_model, [store.get(receipt)], keras_images,
                                             batch_size=keras_batch_size)[0])
        keras_predictions.append(np.stack(folds))
    torch_predictions = []
    for fold in receipts['pytorch']:
        # At most one fold's selected weights live here; never all63 fit states.
        states = [store.get(receipt) for receipt in fold]
        torch_predictions.append(predict_snapshots(torch_model, states, torch_images,
                                                   batch_size=torch_batch_size))
        del states
    result = blend_predictions(np.stack(keras_predictions), average_torch_folds(torch_predictions),
                               constant_images, stage)
    result['checkpoint_counts'] = dict(keras=sum(map(len, receipts['keras'])),
                                        pytorch=sum(map(len, receipts['pytorch'])))
    return result
