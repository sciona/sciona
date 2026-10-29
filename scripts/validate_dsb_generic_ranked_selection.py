"""Qualify generic candidate ranking against hash-verified public hard mining."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import torch
from sciona.atoms.dl.loss import ranked_selection as candidate
from scripts.audit_competition_dsb_semantics import checked_source

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source_root):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('Explicit single-thread environment required')
    torch.set_num_threads(1)
    pins = json.loads((ROOT / 'docs/reviews/competition_dsb_source_pins.json').read_text())
    tree = ast.parse(checked_source(source_root, pins, 'layers.py'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'hard_mining']
    if len(functions) != 1:
        raise ValueError('Source hard-mining inventory changed')
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned-public-hard-mining>', 'exec'), scope)
    cases = 0
    for dtype in (torch.float32, torch.float64):
        for values in ([4., 9., -2., 3., 8.], [2., 0., 2., 2., -1.]):
            for indices in ([4, 0, 3], [3, 0, 2], [], list(range(5))):
                for k in (0, 1, 3, 10):
                    scores = torch.tensor(values, dtype=dtype, requires_grad=True)
                    reference = scores.detach().clone().requires_grad_()
                    candidates = torch.tensor(indices, dtype=torch.int64)
                    expected_scores, expected_indices = scope['hard_mining'](reference[candidates], candidates, k)
                    actual = candidate.highest_scoring_indices(scores, candidates, k)
                    torch.testing.assert_close(actual, expected_indices, rtol=0, atol=0)
                    torch.testing.assert_close(scores[actual], expected_scores, rtol=0, atol=0)
                    actual_gradient, = torch.autograd.grad(scores[actual].sum(), scores)
                    reference_gradient, = torch.autograd.grad(expected_scores.sum(), reference)
                    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)
                    cases += 1
    return dict(passed=True, approved=False, synthetic_only=True, catalog_mutations=0,
                source_commit=pins['commit'], source_hashes={'layers.py': pins['files']['layers.py']},
                cases=cases, exact_indices=True, exact_scores=True, exact_gather_gradients=True,
                native_threads=1, provider_sha256=sha(Path(inspect.getfile(candidate))),
                validator_sha256=sha(Path(__file__)),
                source_extractor_sha256=sha(ROOT / 'scripts/audit_competition_dsb_semantics.py'),
                test_sha256=sha(ROOT / 'tests/test_ranked_selection.py'),
                limitations=['CPU float32/float64 only qualified; ties follow native torch.topk ordering.',
                             'Candidate labels and batch-scaled selection counts remain recipe policy.',
                             'Training integration and catalog approval remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    result = validate(parser.parse_args().source_root)
    (ROOT / 'docs/reviews/dsb_generic_ranked_selection_source_parity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
