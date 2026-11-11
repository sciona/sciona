"""Compare installed CPU optimizer updates with pinned Torch 1.4 classes."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import warnings

import torch
from torch.optim import Optimizer

from sciona.wheat_base_schedule import base_learning_rate
from sciona.legacy_detection_adam import LegacyDetectionAdam


HASHES = dict(sgd='4d5aa550b7820259b9cea34b7a517d4739fa26b3b2199146e7b51bfc12ba6650',
              adam='97067d8382378053cbd13084c84f1da6dac7f584b7fbceff5f61b2070e776091')


def reference(root, name):
    data = (root / (name+'.py')).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[name]:
        raise ValueError('historical optimizer source drift')
    class_name = 'SGD' if name == 'sgd' else 'Adam'
    node = next(n for n in ast.parse(data).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    namespace = dict(torch=torch, Optimizer=Optimizer, math=math, required=object())
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<torch14-'+name+'>', 'exec'), namespace)
    return namespace[class_name]


def main(source, output, implementation):
    torch.set_num_threads(2)
    results = []
    for name in ('sgd', 'adam'):
        old_class = reference(source, name)
        for dtype in (torch.float32, torch.float64):
            rng = torch.Generator().manual_seed(2099)
            initial = [torch.randn(shape, generator=rng, dtype=dtype) for shape in ((17, 19), (19,), (3, 7))]
            a = [torch.nn.Parameter(x.clone()) for x in initial]
            b = [torch.nn.Parameter(x.clone()) for x in initial]
            kwargs = dict(lr=.0005, momentum=.9, weight_decay=.0005) if name == 'sgd' else dict(lr=.00005)
            old = old_class(a, **kwargs)
            new_class = torch.optim.SGD if name == 'sgd' else (torch.optim.Adam if implementation == 'installed' else LegacyDetectionAdam)
            new = new_class(b, **kwargs)
            max_parameter_error = 0.; max_state_error = 0.; exact = True
            for epoch in range(100):
                rate = base_learning_rate(epoch, 'fasterrcnn' if name == 'sgd' else 'effdet')
                old.param_groups[0]['lr'] = new.param_groups[0]['lr'] = rate
                for index, (left, right) in enumerate(zip(a, b)):
                    grad = torch.randn(left.shape, generator=rng, dtype=dtype) * (10. ** (epoch % 7 - 3))
                    left.grad = grad.clone() if (epoch+index) % 11 else None
                    right.grad = grad.clone() if (epoch+index) % 11 else None
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    old.step()
                new.step()
                for left, right in zip(a, b):
                    error = float((left-right).detach().abs().max())
                    max_parameter_error = max(max_parameter_error, error)
                    exact &= torch.equal(left, right)
                    assert old.state[left].keys() == new.state[right].keys()
                    for key, value in old.state[left].items():
                        other = new.state[right][key]
                        if key == 'step':
                            assert value == int(other)
                        else:
                            max_state_error = max(max_state_error, float((value-other).abs().max()))
                            exact &= torch.equal(value, other)
            results.append(dict(optimizer=name, dtype=str(dtype), updates=100,
                                parameters_and_state_exact=bool(exact),
                                maximum_parameter_difference=max_parameter_error,
                                maximum_state_difference=max_state_error))
    report = dict(passed=all(r['parameters_and_state_exact'] for r in results), approved=False,
                  catalog_mutations=0, synthetic_only=True, results=results,
                  source_sha256=HASHES, installed_torch=torch.__version__, adam_implementation=implementation,
                  adam_implementation_sha256=hashlib.sha256(Path('sciona/legacy_detection_adam.py').read_bytes()).hexdigest() if implementation == 'source_order' else None,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limits=['Dense CPU synthetic gradients with source optimizer settings; not detector training.',
                          'Both classes use installed native arithmetic; no historical native-kernel equivalence claim.',
                          'Post-step gradient mutation is excluded: the fit zeros gradients before every backward.',
                          'Apex O1 mixed precision and dynamic loss scaling remain unqualified.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adam-implementation', choices=['installed', 'source_order'], default='installed')
    args = parser.parse_args()
    main(args.source_directory, args.output, args.adam_implementation)
