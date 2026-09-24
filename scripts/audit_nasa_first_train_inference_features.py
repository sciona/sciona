"""Pinned AST comparison of training/submission feature functions and active topology."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='1st Place/Phase 1/'
PINS={
 'training/src/pushback_nasa/pipelines/feature_extraction/nodes.py':'6d725c9b7537e98a734ee79d4856e4f2469a852bab7446b9580910e3d0883729',
 'training/src/pushback_nasa/pipelines/feature_extraction/pipeline.py':'b082e740fde9f2fd03455332e89bda94c30b6b12e19aa11f260478c66c123d0e',
 'training/src/pushback_nasa/pipelines/create_master/nodes.py':'3a4c406754609073c75b67de7832d5eda4f45d20e1081042206ee8a797753061',
 'training/src/pushback_nasa/pipelines/create_master/pipeline.py':'cf83439cfdb3b9e49174a2b3db7d884db46c3d7050cf89d285e48115ac49aba1',
 'submission/utilities.py':'8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'}


def functions(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
def dump(node):return ast.dump(node,include_attributes=False)


def audit(source):
    trees={}
    for path,digest in PINS.items():
        data=(source/BASE/path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Pinned software differs')
        trees[path]=ast.parse(data)
    train=functions(trees['training/src/pushback_nasa/pipelines/feature_extraction/nodes.py'])
    inference=functions(trees['submission/utilities.py'])
    shared=sorted(train.keys()&inference.keys())
    if len(shared)!=7:raise ValueError('Seven shared source families required')
    comparisons=[]
    for name in shared:
        if dump(train[name])!=dump(inference[name]):raise ValueError('Training/inference feature function differs: '+name)
        comparisons.append(dict(function=name,ast_equal=True,ast_sha256=hashlib.sha256(dump(train[name]).encode()).hexdigest()))
    pipeline=functions(trees['training/src/pushback_nasa/pipelines/feature_extraction/pipeline.py'])['create_pipeline']
    assignments={n.targets[0].id:n.value for n in pipeline.body if isinstance(n,ast.Assign)}
    populations=ast.literal_eval(assignments['airports'])
    returned=next(n.value for n in pipeline.body if isinstance(n,ast.Return))
    if not isinstance(returned,ast.Call) or returned.func.id!='pipeline':raise ValueError('Pipeline constructor differs')
    def terms(node):
        if isinstance(node,ast.Name):return [node.id]
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return terms(node.left)+terms(node.right)
        raise ValueError('Unexpected pipeline composition')
    active=terms(returned.args[0]);declared={}
    for name,value in assignments.items():
        if isinstance(value,ast.ListComp):
            if len(value.generators)!=1 or value.generators[0].iter.id!='airports' or value.generators[0].ifs:raise ValueError('Population expansion differs')
            node=value.elt
            declared[name]=next(k.value.id for k in node.keywords if k.arg=='func')
    excluded=sorted(set(declared)-set(active))
    if len(populations)!=10 or len(active)!=8 or len(excluded)!=2:raise ValueError('Active feature topology differs')
    if sorted(declared[n] for n in active if n!='perimeter')!=shared:raise ValueError('Active feature functions differ')
    master=functions(trees['training/src/pushback_nasa/pipelines/create_master/nodes.py'])['build_master']
    arguments=[a.arg for a in master.args.args]
    used={n.id for n in ast.walk(master) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    unused=[name for name in arguments if name not in used]
    if unused!=['tfm','tbfm']:raise ValueError('Unused master inputs differ')
    return dict(passed=True,approved=False,catalog_mutations=0,source_sha256=PINS,shared_functions=comparisons,
        populations=10,active_population_nodes=80,active_feature_nodes=70,perimeter_nodes=10,
        defined_but_excluded_branches=[declared[name] for name in excluded],unused_master_argument_count=2,
        configuration_value_equivalence_verified=False,
        findings=['All seven shared feature functions have identical complete ASTs, including signatures and annotations.',
            'Only perimeter plus seven feature families enter the returned training feature pipeline.',
            'Two other declared branches are excluded, and the corresponding master-function arguments are unused. Historical pipeline wiring still references them; external catalog resolution is not established by this audit.',
            'Corrected shared feature implementations can serve training and inference, but source configuration values and query-population construction need separate qualification.'],
        limitations=['AST equality is structural code evidence, not equality of configuration, training populations, runtime environments or complete fitted outputs.',
            'No complete source graph, corrected raw-feature execution or publication approval claim.'],validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_train_inference_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,identical_feature_functions=7,active_feature_nodes=70,excluded_branches=2,approved=False)))
