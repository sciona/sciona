"""Read pinned private public-software cache; execute selected numerical helpers.

No notebook top-level code, imports, data accesses or outputs are executed.
Source notebooks are not redistributed. This qualifies components only.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from sciona.cassava_ensemble import distribute_unknown
from sciona.cassava_loss import vit_loss


def source_helpers(path, names, namespace):
    source = path.read_text()
    # Notebook magics prevent parsing the full export. Extract only complete
    # top-level definitions by their boundaries, then parse and whitelist them.
    starts = list(re.finditer(r'^(?:def|class) (\w+)[(:]', source, re.M))
    selected = []
    for i, match in enumerate(starts):
        if match.group(1) in names:
            end = starts[i + 1].start() if i + 1 < len(starts) else len(source)
            tree = ast.parse(source[match.start():end])
            node = tree.body[0]
            assert isinstance(node, (ast.FunctionDef, ast.ClassDef))
            assert node.name == match.group(1)
            selected.append(node)
    assert {n.name for n in selected} == names
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<pinned-numerical-helpers>', 'exec'), namespace)
    return namespace


def main():
    cache = Path('/private/tmp/sciona_cassava_winner_source')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    for name in ('inference', 'vit'):
        assert hashlib.sha256((cache / (name + '.py')).read_bytes()).hexdigest() == pins['notebooks'][name]['code_sha256']
    ns = source_helpers(cache / 'vit.py', {
        'log_t', 'exp_t', 'compute_normalization_fixed_point',
        'compute_normalization_binary_search', 'ComputeNormalization',
        'compute_normalization', 'tempered_softmax', 'bi_tempered_logistic_loss',
    }, {'torch': torch})
    mapping = source_helpers(cache / 'inference.py', {'distribute_unknown'}, {'np': np})
    rng = np.random.default_rng(101)
    comparisons = 0
    for dtype in (torch.float32, torch.float64):
        for scale in (.1, 1., 10.):
            x = torch.tensor(rng.normal(size=(20, 5)) * scale, dtype=dtype, requires_grad=True)
            original = x.detach().clone().requires_grad_()
            targets = torch.tensor(rng.dirichlet(np.ones(5), size=20), dtype=dtype)
            expected = ns['bi_tempered_logistic_loss'](original, targets, .8, 1.4, label_smoothing=.06, reduction='none')
            actual = vit_loss(x, targets)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            weights = torch.linspace(.1, 2., 20, dtype=dtype)
            (actual * weights).sum().backward()
            (expected * weights).sum().backward()
            torch.testing.assert_close(x.grad, original.grad, atol=2e-6, rtol=2e-5)
            comparisons += 20
    probabilities = rng.dirichlet(np.ones(6), size=100)
    np.testing.assert_allclose(distribute_unknown(probabilities), mapping['distribute_unknown'](probabilities), atol=1e-15)
    report = dict(approved=False, scope='Pinned numerical components only', synthetic_only=True,
                  notebook_top_level_executed=False, loss_comparisons=comparisons,
                  gradient_coordinates=comparisons * 5, cropnet_mapping_comparisons=100,
                  passed=True)
    report['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path, [
        'sciona/cassava_loss.py', 'sciona/cassava_ensemble.py',
        'scripts/validate_cassava_source_components.py',
        'docs/reviews/competition_cassava_winner_source_pins.json'])}
    Path('docs/reviews/competition_cassava_source_component_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
