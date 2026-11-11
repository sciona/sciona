"""Execute original preparation methods against synthetic in-memory storage."""
import argparse
import ast
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize

from sciona.contrails_grid import create_grid
from sciona.contrails_preparation import augmentation, prepare_training_example


class SyntheticStorage(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def validate(root, source_root):
    torch.set_num_threads(2)
    pins = json.loads((root / 'docs/reviews/competition_contrails_source_pins.json').read_text())
    rng = np.random.default_rng(418)
    thermal = rng.uniform(230, 310, (4, 3, 256, 256)).astype(np.float32)
    label = rng.integers(0, 2, (1, 256, 256)).astype(np.float32)
    mean = rng.uniform(size=label.shape).astype(np.float32)
    storage = SyntheticStorage(x=thermal, y=label, annotation_mean=mean)
    count = 0
    for branch, source in [('single', 'src/unet1024/data.py'), ('temporal', 'src/vit4/data.py')]:
        raw = (source_root / source).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == next(f['sha256'] for f in pins['files'] if f['path'] == source)
        tree = ast.parse(raw)
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['ash_color', 'rescale_range']]
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
        ns = dict(torch=torch, np=np, F=F, h5py=SimpleNamespace(File=lambda *a: storage))
        exec(compile(ast.Module(body=selected + [method], type_ignores=[]), '<pinned-preparation>', 'exec'), ns)
        for mode in [True, False, 'mix']:
            for augmented in [False, True]:
                transform = augmentation() if augmented else None
                source_self = SimpleNamespace(df=SimpleNamespace(iloc=[{'filename': 'synthetic'}]),
                    annotation_mean=mode, resize=Resize(1024 if branch == 'single' else 512, antialias=False),
                    grid=create_grid(512), y_sym_mode='bilinear', augment=transform, augment_prob=.95)
                random.seed(84)
                np.random.seed(17)
                expected = ns['__getitem__'](source_self, 0)
                random.seed(84)
                np.random.seed(17)
                actual = prepare_training_example(thermal, label, mean, branch=branch,
                                                   annotation_mode=mode, augment=transform)
                for key in ['x', 'y', 'y_sym', 'label']:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                assert actual['w'].item() == expected['w'] == (0 if augmented else 1)
                count += 1
    return {'approved': False, 'exact_source_preparation_cases': count,
            'scope': 'Both branches, three annotation modes, source rotation augmentation on/off, entirely synthetic arrays.',
            'hashes': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                       ['sciona/contrails_grid.py', 'sciona/contrails_preparation.py', 'scripts/validate_contrails_preparation.py']}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = validate(root, args.source_root)
    (root / 'docs/reviews/competition_contrails_preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'exact_source_preparation_cases': result['exact_source_preparation_cases']}))
