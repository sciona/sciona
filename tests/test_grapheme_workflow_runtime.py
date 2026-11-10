from pathlib import Path

import numpy as np
import pytest

from scripts.grapheme_synthetic_domains import complete_populations
from sciona.grapheme_workflow_runtime import train_b7


@pytest.mark.parametrize('fault', ['branch', 'targets', 'validation_targets', 'batch', 'normalization', 'extra_policy'])
def test_invalid_training_contract_rejected_before_checkpoint_reads_or_writes(tmp_path, fault):
    data = complete_populations()
    options = dict(epochs=200, batch_size=2, seed=71, mean=[.5]*3, std=[.5]*3,
                   warmup_steps=20, first_schedule_index=0, checkpoint_policy='last')
    branch = 'ood'
    targets = data['ood_training_labels']
    validation_targets = np.array([0, 1, 2], dtype=np.int64)
    if fault == 'branch': branch = 'invalid'
    elif fault == 'targets': targets = np.array([1295]*4)
    elif fault == 'validation_targets': validation_targets = np.array([-1]*3)
    elif fault == 'batch': options['batch_size'] = 3
    elif fault == 'normalization': options['std'] = [0.]*3
    else: options['implicit_policy'] = True
    with pytest.raises(ValueError):
        train_b7(data['hand'], targets, data['calibration'], validation_targets,
            branch=branch, options=options, initialization_path=tmp_path/'absent.pth',
            output_directory=tmp_path/'fit')
    assert list(Path(tmp_path).iterdir()) == []
