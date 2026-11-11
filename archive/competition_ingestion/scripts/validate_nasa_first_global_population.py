"""Compare global feature population construction against pinned source software."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sciona.population_tables import concatenate_labeled_populations

ROOT=Path(__file__).resolve().parents[1]
BASE='1st Place/Phase 1/training/src/pushback_nasa/pipelines/create_master/'
PINS={'nodes.py':'3a4c406754609073c75b67de7832d5eda4f45d20e1081042206ee8a797753061',
      'pipeline.py':'cf83439cfdb3b9e49174a2b3db7d884db46c3d7050cf89d285e48115ac49aba1'}


def validate(source):
    trees={}
    for name,digest in PINS.items():
        data=(source/BASE/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Pinned software changed')
        trees[name]=ast.parse(data)
    function=next(n for n in trees['nodes.py'].body if isinstance(n,ast.FunctionDef) and n.name=='build_global_master')
    labels=ast.literal_eval(next(n.value for n in function.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='airports'))
    select=next(n for n in ast.walk(function) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.BinOp) and isinstance(n.slice.left,ast.List))
    base_columns=ast.literal_eval(select.slice.left)
    markers=[n.left.value for n in ast.walk(select.slice.right) if isinstance(n,ast.Compare) and isinstance(n.left,ast.Constant)]
    label_assignment=next(n for n in ast.walk(function) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript))
    label_column=label_assignment.targets[0].slice.value
    order_column=next(n.args[0].value for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sort_values')
    if len(labels)!=10 or len(base_columns)!=3 or len(markers)!=2:raise ValueError('Source population selection differs')
    pipeline=next(n for n in trees['pipeline.py'].body if isinstance(n,ast.FunctionDef))
    pipeline_labels=ast.literal_eval(next(n.value for n in pipeline.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='airports'))
    if pipeline_labels!=labels:raise ValueError('Global labels differ from source input ordering')
    namespace={'pd':pd}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_global_population','exec'),namespace)
    rng=np.random.default_rng(717)
    compared=0
    for case in range(3):
        tables=[];columns=[]
        for index in range(10):
            n=128
            table=pd.DataFrame({base_columns[0]:np.arange(n)+index*n,order_column:rng.integers(0,16 if case else 100000,size=n),base_columns[2]:rng.uniform(1,100,size=n),
                'synthetic_'+markers[0]:rng.normal(size=n),'synthetic_'+markers[1]:rng.normal(size=n),'ignored_synthetic_feature':rng.normal(size=n)})
            if case==2 and index%2:
                table=table.drop(columns=['synthetic_'+markers[0]])
            tables.append(table)
            columns.append(base_columns+[c for c in table.columns if any(marker in c for marker in markers)])
        originals=[t.copy(deep=True) for t in tables]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.SettingWithCopyWarning)
            expected=namespace['build_global_master'](*[t.copy(deep=True) for t in tables])
        actual=concatenate_labeled_populations(tables,labels,columns,label_column,order_column)
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        for before,after in zip(originals,tables):pd.testing.assert_frame_equal(before,after,check_exact=True)
        if len(actual)!=1280:raise ValueError('Rows lost or duplicated')
        compared+=len(actual)
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,populations=10,cases=3,exact_compared_rows=compared,
        source_sha256=PINS,source_population_order_verified=True,
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['sciona/population_tables.py','tests/test_population_tables.py','scripts/validate_nasa_first_global_population.py']},
        scope='Global selected-table concatenation and source population ordering only; input features are synthetic and not produced by the complete feature pipeline.',
        limitations=['Column inclusion is source-adapter logic; reusable concatenation receives explicit per-population selections and labels.',
            'Source pandas quicksort behavior retained, including ties; no stable cross-version tie-order guarantee.',
            'Different selected feature sets union with missing values; no imputation or training eligibility is inferred.',
            'Native global categorical training and all 21 fits remain separate qualifications.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_global_population.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','cases','exact_compared_rows','approved']}))
