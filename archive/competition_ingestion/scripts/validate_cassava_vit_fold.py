"""Full ten-epoch ViT CPU diagnostic with explicit population correction."""
import hashlib
import json
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from sciona.cassava_vit_fold import train_fold


def main():
    torch.set_num_threads(4)
    torch.manual_seed(73); random.seed(73); np.random.seed(73)
    def jpeg(value):
        ok, encoded = cv2.imencode('.jpg', np.full((600, 800, 3), value, np.uint8))
        assert ok
        return encoded.tobytes()
    with tempfile.TemporaryDirectory(prefix='cassava-vit-fold-') as temp:
        model, history, checkpoint, updates = train_fold([jpeg(31), jpeg(79)], [1, 1], [jpeg(113)], [1],
            pretrained=False, batch_size=1, workers=0, output_directory=Path(temp) / 'fold')
        assert len(history) == 10 and updates == 10
        assert np.isfinite([[h[k] for k in ['lr', 'last_validation_batch_loss', 'accuracy']] for h in history]).all()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        scores = [h['accuracy'] for h in history]
        best = max(i for i, score in enumerate(scores) if score == max(scores))
        assert saved['epoch'] == best
        for name, tensor in model.state_dict().items(): torch.testing.assert_close(tensor, saved['model'][name], atol=0, rtol=0)
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=1e-4 / 7)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4, last_epoch=-1)
        for record in history:
            assert record['lr'] == optimizer.param_groups[0]['lr']
            scheduler.step()
        report = {'approved': False, 'passed': True, 'synthetic_only': True, 'epochs_executed': 10,
            'accumulated_optimizer_steps': updates, 'diagnostic_batch_size': 1, 'workers': 0,
            'pretrained_weights_used': False, 'latest_best_epoch': best, 'checkpoint_weights_verified': True,
            'scheduler_history_replayed': True,
            'explicit_correction': 'Validation is not concatenated into training; pinned notebook does concatenate it.',
            'scope': 'Full-resolution CPU lifecycle diagnostic; real fold provenance, original batch/workers, pretrained and TPU parity unqualified.'}
    paths = ['sciona/cassava_pretrained.py', 'sciona/cassava_vit_fold.py', 'sciona/cassava_vit_model.py', 'sciona/cassava_vit_images.py',
             'sciona/cassava_vit_ordered_loss.py', 'sciona/cassava_resnext_fold.py', 'sciona/cassava_resnext_images.py',
             'scripts/validate_cassava_vit_fold.py']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_vit_fold_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
