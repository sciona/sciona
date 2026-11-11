"""Reviewed fit/checkpoint identities for all three TGS ensemble rounds."""
import numpy as np


def ensemble_selection(stage):
    if type(stage) is not int or stage not in (1, 2, 3):
        raise ValueError('stage must be 1, 2 or 3')
    keras_round = 1 if stage == 1 else 2
    phases = (4, 3, 2, 1) if stage == 1 else (5, 4, 3, 2)
    torch_phase = {1: 0, 2: 1, 3: 3}[stage]
    keras = [[dict(fit=f'keras.r{keras_round}.p{phase}.f{fold}', selection='best_validation')
              for fold in range(5)] for phase in phases]
    torch = []
    for fold in range(5):
        indices = list(range(1, 6)) if stage == 1 else list(range(4 if stage == 2 or fold < 3 else 3))
        torch.append(dict(fit=f'torch.p{torch_phase}.f{fold}', selection='cycle_best', indices=indices))
    return dict(stage=stage, keras=keras, pytorch=torch,
                keras_weights=[1, 1, 1, 1 if stage == 1 else 3],
                branch_weights=[1, 1], torch_fold_weights=[1] * 5,
                produces_pseudo_round=stage if stage < 3 else None,
                mosaic_postprocessing=stage == 3)


def average_torch_folds(predictions):
    """Equal fold weights even when folds select unequal snapshot counts."""
    if len(predictions) != 5:
        raise ValueError('exactly five fold predictions required')
    values = [np.asarray(p, dtype=np.float64) for p in predictions]
    if (values[0].ndim != 3 or any(size == 0 for size in values[0].shape)
            or any(v.shape != values[0].shape or not np.isfinite(v).all() for v in values)):
        raise ValueError('aligned finite NHW fold predictions required')
    # The source adds the five fold arrays in order before dividing.
    return (values[0] + values[1] + values[2] + values[3] + values[4]) / 5
