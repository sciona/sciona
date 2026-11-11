"""Run all v43 folds using full models and entirely synthetic populations."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import sciona.contrails_lifecycle as lifecycle


def validate(root):
    torch.set_num_threads(2)
    torch.manual_seed(627)
    np.random.seed(4)
    random.seed(627)
    rng = np.random.default_rng(627)

    def example():
        return dict(thermal=rng.uniform(230, 310, (4, 3, 256, 256)).astype(np.float32),
                    label=rng.integers(0, 2, (1, 256, 256)).astype(np.float32),
                    annotation_mean=rng.uniform(size=(1, 256, 256)).astype(np.float32))

    training = [example() for _ in range(10)]
    scoring = [example()]
    prediction = [example()['thermal']]
    original = lifecycle.train_fold
    completed = []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(sum(e['optimizer_steps'] for e in result['epochs']))
        print(json.dumps({'trained_folds': len(completed), 'last_fold_updates': completed[-1]}), flush=True)
        return result

    start = time.monotonic()
    lifecycle.train_fold = observed
    try:
        result = lifecycle.run_lifecycle(training, scoring, prediction, initialization='random',
                                         variant='v43', batch_size=1, num_workers=0, epoch_limit=1)
    finally:
        lifecycle.train_fold = original
    assert [(f['branch'], f['fold']) for f in result['folds']] == [
        ('temporal', 5), ('temporal', 7), ('single', 3), ('single', 4)]
    assert completed == [3, 3, 3, 3]
    assert result['probabilities'].shape == (1, 1, 256, 256)
    assert np.isfinite(result['probabilities']).all() and len(result['masks']) == 1
    paths = ['sciona/contrails_lifecycle.py', 'sciona/contrails_fold.py',
             'sciona/contrails_defaults.json', 'scripts/validate_contrails_lifecycle.py']
    return {'approved': False, 'checks': {'full_model_folds': 4, 'optimizer_updates': sum(completed),
                                        'terminal_checkpoints_used_for_inference': 4,
                                        'ensembled_predictions': 1},
            'elapsed_seconds': time.monotonic() - start,
            'hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
            'scope': 'All v43 folds, source-resolution models; synthetic distinct training/scoring/prediction populations. Explicit random init, one epoch without schedule rescaling, batch1/workers0. Full default schedule and pretrained performance not claimed.'}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = validate(root)
    (root / 'docs/reviews/competition_contrails_lifecycle.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['checks']), flush=True)
