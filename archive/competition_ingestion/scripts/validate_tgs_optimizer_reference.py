"""Compare configured RMSprop with externally supplied Keras2.2.0 expressions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sciona.tgs_training_state import optimizer_for


def main(source, output):
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == '3d26c69b2b8d088fe5bd8b6f5d4daf04dd56d481f0ad376964e3b10c50747589'
    cls = next(n for n in ast.parse(data).body if isinstance(n, ast.ClassDef) and n.name == 'RMSprop')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_updates')
    loop = next(n for n in method.body if isinstance(n, ast.For))
    # Only direct loop assignments; the nested optional constraint branch is
    # outside the winner's unconstrained RMSprop configuration.
    formulas = {n.targets[0].id: compile(ast.Expression(n.value), '<pinned-rmsprop-formula>', 'eval')
                for n in loop.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('new_a', 'new_p')}
    assert set(formulas) == {'new_a', 'new_p'}
    model = torch.nn.Linear(7, 1, bias=False)
    initial = np.linspace(-.4, .6, 7, dtype=np.float32)[None, :]
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(initial))
    optimizer = optimizer_for(model, 'keras', .0001)
    value, accumulator, errors = initial.copy(), np.zeros_like(initial), []
    for step in range(16):
        gradient = np.array([[0., 1e-12, -1e-8, .01, -.2, 3., -7.]], dtype=np.float32) * ((-1)**step)
        rate = np.float32(.0001 if step < 8 else .000025)
        namespace = dict(self=SimpleNamespace(rho=np.float32(.9), epsilon=1e-7),
                         K=SimpleNamespace(square=np.square, sqrt=np.sqrt),
                         p=value, a=accumulator, g=gradient, lr=rate)
        accumulator = eval(formulas['new_a'], {'__builtins__': {}}, namespace)
        namespace['new_a'] = accumulator
        value = eval(formulas['new_p'], {'__builtins__': {}}, namespace)
        model.weight.grad = torch.from_numpy(gradient.copy())
        optimizer.param_groups[0]['lr'] = float(rate)
        optimizer.step()
        actual = model.weight.detach().numpy()
        np.testing.assert_allclose(actual, value, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(optimizer.state[model.weight]['square_avg'].numpy(), accumulator, rtol=2e-6, atol=1e-12)
        errors.append(float(np.max(np.abs(actual - value))))
    report = dict(passed=True, source_url='https://raw.githubusercontent.com/keras-team/keras/2.2.0/keras/optimizers.py',
                  source_sha256=digest, steps=16, parameter_max_error=max(errors), catalog_mutations=0,
                  verified=['Actual pinned Keras2.2.0 RMSprop update expressions evaluated with numeric backend.',
                            'Zero, tiny, large and alternating gradients plus learning-rate transition.',
                            'Parameters and running-square accumulators compared to implemented optimizer.'],
                  limits=['Dense unconstrained float32 RMSprop configuration only; not full historical TensorFlow runtime equivalence.'],
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
