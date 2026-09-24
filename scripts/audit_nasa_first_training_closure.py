"""Read-only AST audit of pinned first-place model production and consumption."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = '1st Place/Phase 1/training/src/pushback_nasa/'
PINS = {
    BASE + 'pipeline_registry.py': '75645934d354b4637d5613a072e08720ac463dbae2382ded8ad9a7070727b5e3',
    BASE + 'pipelines/train_models/pipeline.py': 'c410432dc7466851ce9650c4e5c141031f83bc80cdf743f38cb323b3eb0cf99a',
    BASE + 'pipelines/generate_predictions/pipeline.py': 'fa5c9896ec542febbbedc8751abc6c4bdf51b8742a5b05fbc8eeeb8b778d134c',
    BASE + 'pipelines/train_models/nodes.py': '9b96dd580d39b9ae6126c380c390f57c21b10d3cbc7d028af5a6639799da271d',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def function(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(matches) == 1, 'Unique function required: ' + name)
    return matches[0]


def render(node, population):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                pieces.append(value.value)
            else:
                require(isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name)
                        and value.value.id == 'airport' and value.format_spec is None and value.conversion == -1,
                        'Unexpected source interpolation')
                pieces.append(population)
        return ''.join(pieces)
    raise ValueError('Unsupported source reference')


def audit(source):
    trees = {}
    for path, digest in PINS.items():
        data = (source / path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, 'Pinned source differs: ' + path)
        trees[path] = ast.parse(data)
    train = function(trees[BASE + 'pipelines/train_models/pipeline.py'], 'create_pipeline')
    assignments = {n.targets[0].id: n.value for n in train.body if isinstance(n, ast.Assign)}
    populations = ast.literal_eval(assignments['airports'])
    require(len(populations) == len(set(populations)) == 10, 'Ten unique source populations required')
    produced = set()
    fits = {}
    for group, expected_trainer, width in [('models_v02', 'train_catboost_diff', 3), ('models_v1', 'train_catboost', 1), ('global_model', 'train_catboost', 1)]:
        value = assignments[group]
        if isinstance(value, ast.ListComp):
            require(len(value.generators) == 1, 'One source population loop required')
            loop = value.generators[0]
            require(ast.dump(loop.target) == ast.dump(ast.Name(id='airport', ctx=ast.Store()))
                    and isinstance(loop.iter, ast.Name) and loop.iter.id == 'airports'
                    and not loop.ifs and not loop.is_async, 'Population expansion differs')
            node, slots = value.elt, populations
        else:
            require(isinstance(value, ast.List) and len(value.elts) == 1, 'One global fit required')
            node, slots = value.elts[0], ['']
        require(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'node', 'Pipeline node required')
        kwargs = {k.arg: k.value for k in node.keywords}
        require(isinstance(kwargs['func'], ast.Name) and kwargs['func'].id == expected_trainer, 'Training function differs')
        outputs = kwargs['outputs'].elts if isinstance(kwargs['outputs'], ast.List) else [kwargs['outputs']]
        require(len(outputs) == width, 'Output arity differs')
        for population in slots:
            produced.update(render(output, population) for output in outputs)
        fits[group] = len(slots)
    returned = next(n.value for n in train.body if isinstance(n, ast.Return))
    require(ast.dump(returned) == ast.dump(ast.parse('pipeline(models_v02 + models_v1 + global_model)', mode='eval').body), 'Training pipeline composition differs')
    trainer = function(trees[BASE + 'pipelines/train_models/nodes.py'], 'train_catboost_diff')
    returned = next(n.value for n in trainer.body if isinstance(n, ast.Return))
    require(isinstance(returned, ast.Tuple) and [n.id for n in returned.elts] == ['best_model', 'best_model', 'feat_imps'], 'Residual model alias differs')
    prediction = function(trees[BASE + 'pipelines/generate_predictions/pipeline.py'], 'create_pipeline')
    calls = [n for n in ast.walk(prediction) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'node']
    retrieval = [n for n in calls if any(k.arg == 'func' and isinstance(k.value, ast.Name) and k.value.id == 'retrieve_predictions' for k in n.keywords)]
    require(len(retrieval) == 1, 'One population prediction template required')
    inputs = next(k.value for k in retrieval[0].keywords if k.arg == 'inputs')
    require(isinstance(inputs, ast.List) and len(inputs.elts) == 4, 'Prediction input arity differs')
    consumed = {render(inputs.elts[2], population) for population in populations}
    missing = consumed - produced
    require(len(missing) == 10, 'Expected unresolved model-name wiring differs')
    return dict(passed=True,approved=False,catalog_mutations=0,source_commit='09f2f5d2940dd6b63b93115a0e9fb9e8a964c700',
        software_sha256=PINS,populations=10,training_fit_nodes=sum(fits.values()),residual_fits=10,local_direct_fits=10,global_direct_fits=1,
        model_output_slots=31,feature_importance_output_slots=10,residual_alias_pairs=10,
        downstream_model_references_not_produced=10,
        findings=['The residual trainer returns the same fitted object to two model output slots per population; 31 model slots do not imply 31 independent fits.',
                  'A global direct-target fit is explicitly wired in the training pipeline.',
                  'The retained prediction pipeline requests ten unsuffixed model names absent from the training outputs. External catalog aliases or corrected wiring must be established.'],
        remaining=['Establish upstream global population construction and any external catalog aliases.',
                   'Recover exact inference ensemble, residual offsets, clipping and population-specific branches.',
                   'Implement reusable training/state/routing operations with synthetic parity and at most four total workers; source thread_count=-1 must be bounded explicitly.',
                   'Complete native execution, licensing, source-intake correction and Tier 3 publication gates.'],
        scope='Static software AST and exact source-byte audit only. No fit, runtime parity, upstream population closure or full-solution approval claim. No source dataset records or identifying metadata included.',
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_training_closure.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','training_fit_nodes','model_output_slots','residual_alias_pairs','downstream_model_references_not_produced','approved']}))
