"""Round-trip synthetic checkpoints through all four actual legacy backbones."""
import gc
import hashlib
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pretrainedmodels
import torch

from sciona.aptos_initialization import build_from_checkpoint
from sciona.aptos_models import FAMILIES


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(719)
    rows = {}
    with TemporaryDirectory(prefix='aptos-synthetic-init-') as directory:
        for family, (factory, pool, _, _) in FAMILIES.items():
            classes = 1001 if family.startswith('inception_') else 1000
            with patch('torch.utils.model_zoo.load_url', side_effect=AssertionError('Implicit download')):
                original = getattr(pretrainedmodels, factory)(num_classes=classes, pretrained=None)
                state = original.state_dict()
                expected = {k: tensor_digest(v) for k, v in state.items() if not k.startswith('last_linear.')}
                checkpoint = Path(directory) / 'synthetic.pt'
                torch.save(state, checkpoint)
                with checkpoint.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                del state, original
                gc.collect()
                model = build_from_checkpoint(family, checkpoint, expected_sha256=digest)
            actual = {k: tensor_digest(v) for k, v in model.state_dict().items()
                      if not k.startswith('last_linear.') and k != pool + '.p'}
            assert actual == expected
            assert model.last_linear.out_features == 1
            assert getattr(model, pool).p.item() == 3.0
            source = Path(inspect.getfile(getattr(pretrainedmodels, factory)))
            rows[family] = {'matched_backbone_tensors': len(expected),
                            'original_classifier_rows': classes,
                            'backbone_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            del model
            checkpoint.unlink()
            gc.collect()
            print('PASS synthetic checkpoint roundtrip', family, flush=True)
    paths = ['sciona/aptos_initialization.py', 'sciona/aptos_models.py',
             'sciona/aptos_pooling.py', 'scripts/validate_aptos_initialization.py',
             'tests/test_aptos_initialization.py']
    report = {'passed': True, 'approved': False, 'catalog_mutations': 0,
              'synthetic_only': True, 'pretrained_weights_used': False,
              'families': rows, 'temporary_checkpoints_removed': True,
              'scope': 'Actual legacy backbone state roundtrip; not publisher weight provenance, training or full-workflow qualification.',
              'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_aptos_initialization_validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
