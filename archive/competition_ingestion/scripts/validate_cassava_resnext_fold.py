"""Actual 15-epoch synthetic full-resolution fold and best-checkpoint replay."""
import hashlib
import json
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from sciona.cassava_resnext import validate
from sciona.cassava_resnext_fold import train_fold, EncodedPopulation, scheduler_for


def main():
    torch.set_num_threads(4)
    torch.manual_seed(72); random.seed(72); np.random.seed(72)
    def encoded(value):
        ok, blob = cv2.imencode('.jpg', np.full((600, 800, 3), value, np.uint8))
        assert ok
        return blob.tobytes()
    training = [encoded(51), encoded(97)]
    validation = [encoded(71)]
    with tempfile.TemporaryDirectory(prefix='cassava-resnext-fold-') as temp:
        model, history, checkpoint = train_fold(training, [1, 1], validation, [1], pretrained=False,
            batch_size=1, workers=0, output_directory=Path(temp) / 'fold')
        assert len(history) == 15
        assert np.isfinite([[h[k] for k in ['loss', 'val_loss', 'accuracy', 'lr']] for h in history]).all()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        best = int(np.argmax([h['accuracy'] for h in history]))
        assert saved['epoch'] == best
        _, probabilities = validate(model, DataLoader(EncodedPopulation(validation, [1], training=False), batch_size=1))
        np.testing.assert_allclose(probabilities, saved['probabilities'].numpy(), atol=1e-7)
        # Replay scheduler independently on recorded validation losses.
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=1e-4)
        scheduler = scheduler_for(optimizer)
        for h in history:
            assert optimizer.param_groups[0]['lr'] == h['lr']
            scheduler.step(h['val_loss'])
        # Patience means six bad epochs after the first good observation.
        parameter = torch.nn.Parameter(torch.zeros(()))
        opt = torch.optim.Adam([parameter], lr=1e-4)
        schedule = scheduler_for(opt)
        for _ in range(6): schedule.step(1.)
        assert opt.param_groups[0]['lr'] == 1e-4
        schedule.step(1.)
        assert np.isclose(opt.param_groups[0]['lr'], 2e-5)
        report = {'approved': False, 'passed': True, 'synthetic_only': True,
            'epochs_executed': 15, 'optimizer_steps': 30, 'diagnostic_batch_size': 1, 'workers': 0,
            'pretrained_weights_used': False, 'strict_best_epoch': best, 'checkpoint_prediction_replay': True,
            'scheduler_history_replayed': True, 'patience_boundary_verified': True,
            'scope': 'Full-resolution CPU fold diagnostic; original batch/workers, pretrained provenance and complete ensemble remain unqualified.'}
    paths = ['sciona/cassava_pretrained.py', 'sciona/cassava_resnext_fold.py', 'sciona/cassava_resnext.py', 'sciona/cassava_resnext_images.py', 'scripts/validate_cassava_resnext_fold.py']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_resnext_fold_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
