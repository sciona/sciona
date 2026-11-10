"""Extract declared model budgets without importing either legacy framework."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def value(node, variables):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [value(n, variables) for n in node.elts]
    if isinstance(node, ast.BinOp):
        left, right = value(node.left, variables), value(node.right, variables)
        if isinstance(node.op, ast.Pow):
            return left**right
    # Preserve function references and conditionals as source expressions;
    # this extractor does not execute them or silently resolve new policies.
    return dict(source_expression=ast.unparse(node))


def inventory(directory):
    triage = json.loads((ROOT/'docs/reviews/competition_mercari_source_triage.json').read_text())
    pins = {r['path']: r['sha256'] for r in triage['source_files']}
    names = ['mercari/main_tf.py', 'mercari/main_mx.py', 'script_template.tmpl']
    for name in names:
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != pins[name]:
            raise ValueError('Pinned source changed: '+name)
    variants = {}
    for backend, name in [('tensorflow', names[0]), ('mxnet', names[1])]:
        tree = ast.parse((directory/name).read_text())
        models = []
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('define_models_')]
        if [n.name for n in functions] != ['define_models_1', 'define_models_2', 'define_models_3']:
            raise ValueError('Expected three ordered model families')
        for group, function in enumerate(functions, 1):
            assignments = {t.id: n.value for n in function.body if isinstance(n, ast.Assign)
                           for t in n.targets if isinstance(t, ast.Name)}
            variables = dict(h0=value(assignments['h0'], {}), n_epoch=3)
            if ast.unparse(assignments['n_epoch']) != '3 if TEST_SIZE == 1 else 1':
                raise ValueError('Source epoch policy changed')
            loop = next(n for n in function.body if isinstance(n, ast.For))
            if (ast.unparse(loop.iter) != 'enumerate(models, seed * 100)'
                    or len(loop.body) != 1 or ast.unparse(loop.body[0]) != 'model.seed = i'):
                raise ValueError('Source seed assignment changed')
            declared = assignments['models']
            if not isinstance(declared, ast.List) or len(declared.elts) != 4:
                raise ValueError('Four models per declared family required')
            for ordinal, constructor in enumerate(declared.elts):
                if not isinstance(constructor, ast.Call) or not isinstance(constructor.func, ast.Name) or constructor.args:
                    raise ValueError('Explicit named model construction required')
                models.append(dict(group=group, ordinal=ordinal, model_class=constructor.func.id,
                    seed=group*100+ordinal,
                    explicit_controls={k.arg: value(k.value, variables) for k in constructor.keywords}))
        variants[backend] = models
    launch = (directory/'script_template.tmpl').read_text()
    commands = [line for line in launch.splitlines() if line.startswith("run('python mercari/")]
    expected = [f"run('python mercari/main_{{name}}.py {action}')" for group in (1, 3, 2) for action in (group, 'merge')]
    if commands != expected:
        raise ValueError('Source group/blend ordering changed')
    return dict(format='mercari-declared-model-inventory.v1', approved=False, catalog_mutations=0,
        source_commit=triage['source_commit'], source_sha256={n: pins[n] for n in names},
        variants=variants, group_execution_order=[1, 3, 2], refit_blend_after_each_group=True,
        declared_models_per_backend=12, full_input_epochs_per_model=3, reduced_input_epochs_per_model=1,
        scope='Static inventory of explicit constructor overrides, not resolved inherited defaults or full winning model inventory.',
        limitations=['Select and qualify a backend; TensorFlow and MXNet models are not assumed equivalent.',
            'Additional four best-submission models noted by MXNet source lack exact pinned seed/configuration evidence.',
            'TensorFlow Huber-named models use the inherited MSE builder in this commit.',
            'No data or training executed. Parameter extraction is not source-runtime parity or publication approval.',
            'Caller must limit parallel fitting and native pools to the shared four-core budget; source parallel defaults are not authorized.'],
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    report = inventory(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_mercari_model_inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(backends=list(report['variants']), declared_models_per_backend=12,
                         group_execution_order=report['group_execution_order'], approved=False)))
