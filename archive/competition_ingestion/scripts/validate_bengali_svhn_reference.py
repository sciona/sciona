"""Compare all SVHN subpolicies against the hash-pinned winner-linked source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageOps

from sciona.svhn_autoaugment import SVHNAutoAugment


SOURCE_SHA256 = '41099a6a0157d1e1cb6b509f94d3ba88297fd61283a461a610b5e24d1926aae8'
COMMIT = '17d718251f25c0d9413bf30f91b523907924f33a'


class ForcedDraws:
    def __init__(self, index, first, second, sign):
        self.index, self.values, self.sign = index, iter((first, second)), sign
        self.calls = []

    def randint(self, low, high):
        assert (low, high) == (0, 24)
        self.calls.append(('index', self.index))
        return self.index

    def random(self):
        value = next(self.values)
        self.calls.append(('probability', value))
        return value

    def choice(self, values):
        assert values == [-1, 1]
        self.calls.append(('sign', self.sign))
        return self.sign


def main(reference, output):
    raw = reference.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('SVHN reference source content changed')
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('SVHNPolicy', 'SubPolicy')]
    if len(nodes) != 2:
        raise ValueError('reference classes absent')
    # np.int was the builtin int alias in the source environment. Supply it
    # only to the isolated reference namespace; do not mutate installed NumPy.
    namespace = dict(np=SimpleNamespace(linspace=np.linspace, round=np.round, int=int),
                     Image=Image, ImageEnhance=ImageEnhance, ImageOps=ImageOps)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-svhn-reference>', 'exec'), namespace)
    rows, columns = np.indices((137, 236))
    pixels = np.stack([(columns * 7 + rows * 3) % 256, (columns * 11 + rows * 13) % 256,
                       ((columns // 7 + rows // 5) % 2) * 255], axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels)
    cases = 0
    # Exercise both probability decisions and signs for every subpolicy,
    # including the always-disabled operation with probability zero.
    for fill in [(128, 128, 128), (19, 83, 211)]:
        original = namespace['SVHNPolicy'](fillcolor=fill)
        candidate = SVHNAutoAugment(seed=19, fillcolor=fill)
        for index in range(25):
            for first in (0., 1.):
                for second in (0., 1.):
                    for sign in (-1, 1):
                        a, b = [ForcedDraws(index, first, second, sign) for _ in range(2)]
                        namespace['random'], candidate.rng = a, b
                        expected, actual = np.asarray(original(image)), np.asarray(candidate(image))
                        np.testing.assert_array_equal(actual, expected)
                        assert a.calls == b.calls
                        cases += 1
    original = namespace['SVHNPolicy']()
    candidate = SVHNAutoAugment(seed=174)
    namespace['random'] = random.Random(174)
    for _ in range(128):
        np.testing.assert_array_equal(np.asarray(candidate(image)), np.asarray(original(image)))
        assert candidate.rng.getstate() == namespace['random'].getstate()
    np.testing.assert_array_equal(np.asarray(image), pixels)
    files = ['sciona/svhn_autoaugment.py', 'scripts/validate_bengali_svhn_reference.py',
             'docs/licenses/deepvoltaire-autoaugment-LICENSE.txt']
    report = dict(passed=True, synthetic_only=True, approved=False, catalog_mutations=0,
        source_url='https://github.com/DeepVoltaire/AutoAugment/blob/' + COMMIT + '/autoaugment.py',
        source_commit=COMMIT, source_sha256=SOURCE_SHA256, source_license='MIT',
        exhaustive_subpolicy_cases=cases, seeded_sequence_cases=128,
        maximum_pixel_difference=0, random_draws_and_state_exact=True,
        versions={'pillow': PIL.__version__, 'numpy': np.__version__},
        implementation_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Revision predates the winner writeup; the winner did not pin an exact revision.',
                'Reference and reconstruction share installed Pillow; historical raster-library equivalence remains unqualified.',
                'Only SVHN augmentation is qualified here, not B7 training, normalization, warmup or complete pipeline publication.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'implementation_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.reference, args.output)
