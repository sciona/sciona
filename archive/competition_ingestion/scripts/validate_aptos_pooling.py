"""Compare local GeM to pinned source, including input and exponent gradients."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from sciona.aptos_pooling import GeM


def main():
    torch.set_num_threads(2)
    pins_path = Path('docs/reviews/competition_aptos_pooling_source_pins.json')
    pins = json.loads(pins_path.read_text())
    directory = Path('/private/tmp/sciona_aptos_pooling_reference')
    namespace = dict(torch=torch, nn=nn, F=F, Parameter=nn.Parameter)
    for name, symbol in [('cirtorch/layers/functional.py', 'gem'), ('cirtorch/layers/pooling.py', 'GeM')]:
        source = (directory / name.replace('/', '_')).read_bytes()
        if hashlib.sha256(source).hexdigest() != pins['files'][name]['sha256']:
            raise ValueError('Pinned pooling source differs')
        selected = [node for node in ast.parse(source).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol]
        if len(selected) != 1:
            raise ValueError('Unique source definition required')
        exec(compile(ast.Module(body=selected, type_ignores=[]), name, 'exec'), namespace)
        if symbol == 'gem':
            namespace['LF'] = SimpleNamespace(gem=namespace['gem'])
    reference = namespace['GeM']
    cases, updates = 0, 0
    for dtype in [torch.float32, torch.float64]:
        for exponent in [1., 3., 4.5]:
            for shape in [(1, 2, 2, 3), (2, 3, 3, 4)]:
                values = torch.linspace(-.1, 2., steps=math_prod(shape), dtype=dtype).reshape(shape)
                local, source = GeM(exponent).to(dtype), reference(exponent).to(dtype)
                optimizers = [torch.optim.SGD(model.parameters(), lr=.01) for model in [local, source]]
                for _ in range(3):
                    x, y = values.clone().requires_grad_(), values.clone().requires_grad_()
                    a, b = local(x), source(y)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    a.sum().backward(); b.sum().backward()
                    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                    torch.testing.assert_close(local.p.grad, source.p.grad, rtol=0, atol=0)
                    for optimizer in optimizers:
                        optimizer.step(); optimizer.zero_grad()
                    torch.testing.assert_close(local.p, source.p, rtol=0, atol=0)
                    updates += 1
                cases += 1
    paths = ['sciona/aptos_pooling.py', 'tests/test_aptos_pooling.py',
             'scripts/validate_aptos_pooling.py', str(pins_path), 'docs/licenses/APTOS-pooling-MIT.txt']
    report = {'approved': False, 'catalog_mutations': 0, 'passed': True,
        'synthetic_only': True, 'source_cases': cases, 'paired_optimizer_updates': updates,
        'outputs_input_gradients_parameter_gradients_exact': True,
        'runtime': {'torch': torch.__version__},
        'scope': 'Pooling component only; full APTOS model training, refinement and ensemble remain unqualified. Learned exponent is unconstrained as in source.',
        'sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}}
    Path('docs/reviews/competition_aptos_pooling_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', cases, 'source cases and', updates, 'paired optimizer updates')


def math_prod(shape):
    import math
    return math.prod(shape)


if __name__ == '__main__':
    main()
