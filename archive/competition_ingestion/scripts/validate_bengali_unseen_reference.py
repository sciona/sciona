"""Compare joint-logit marginalization with the pinned submission function."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sciona.bengali_unseen_ensemble import marginal_predictions


SOURCE_SHA256 = 'c060484bb402246cb592fb8dd3027431fc441a4cf07d462b2065c1264f310383'


def main(source, output):
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError('pinned submission source required')
    # Shell/magic notebook setup is not Python and is never executed here.
    python_source = '\n'.join('' if line.lstrip().startswith(('!', '%')) else line
                              for line in payload.decode().splitlines())
    tree = ast.parse(python_source)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_merc')
    function = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == 'out2pred')
    # The display-only glyph string is replaced by its numeric components.
    # It has no role in source probabilities or prediction selection.
    namespace = dict(torch=torch, nn=nn, np=np, NUM_GRAPHEME_ROOT=168,
                     NUM_VOWEL_DIACRITIC=11, NUM_CONSONANT_DIACRITIC=8,
                     label_to_grapheme=lambda g, v, c: (int(g), int(v), int(c)))
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned-out2pred>', 'exec'), namespace)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(2053)
    cases = []
    for index in range(64):
        shape = (1 + index % 4, 14784)
        scale = (0., .01, 1., 50.)[index % 4]
        cases.append((torch.randn(shape, generator=generator) * scale,
                      torch.randn(shape, generator=generator) * scale))
    # Exercise disagreement between joint argmax and independent marginals.
    special = torch.full((1, 14784), -100.)
    special[0, [0, 96, 89]] = torch.tensor([.4, .35, .25]).log()
    cases.append((special, special.clone()))
    last = torch.full((1, 14784), -100.)
    last[0, -1] = 0
    cases.append((last, last.clone()))
    for a, b in cases:
        expected = namespace['out2pred']((a, b))
        actual = marginal_predictions(a, b)
        components = torch.tensor([[r['grapheme_root'], r['vowel_diacritic'],
                                    r['consonant_diacritic']] for r in expected])
        torch.testing.assert_close(actual['components'], components, rtol=0, atol=0)
        torch.testing.assert_close(actual['joint_classes'],
                                   torch.tensor([r['pred'] for r in expected]), rtol=0, atol=0)
    root = Path(__file__).resolve().parents[1]
    files = ['sciona/bengali_unseen_ensemble.py', 'scripts/validate_bengali_unseen_reference.py']
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
                  source_sha256=SOURCE_SHA256, cases=len(cases),
                  examples=sum(len(a) for a, _ in cases), joint_and_component_predictions_exact=True,
                  implementation_sha256={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
                  limits=['Pinned reduced-notebook out2pred only; no trained two-font or full routing execution.',
                          'Source glyph display replaced with numeric components; no glyph mapping qualification.',
                          'Installed Torch softmax and reduction kernels shared by both executions.',
                          'Submission remapping is covered separately and is not checked by out2pred.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
