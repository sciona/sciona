"""Pinned fold partition/inventory oracles using synthetic positional records."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_folds import assign_folds,eligible_crop_positions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    p=args.source_root/'preprocessing/generate_folds.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==next(r['sha256'] for r in pins['files'] if r['path']=='preprocessing/generate_folds.py')
    tree=ast.parse(p.read_text());main_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start=next(i for i,n in enumerate(main_fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sz' for t in n.targets))
    assert isinstance(main_fn.body[start+2],ast.For)
    code=compile(ast.Module(body=main_fn.body[start:start+3],type_ignores=[]),'<source-fold-partitions>','exec')
    source_cases=0
    for splits in (1,2,3,7,10,16,49,50):
        namespace={'args':SimpleNamespace(n_splits=splits)};exec(code,namespace)
        expected=[next(i for i,group in enumerate(namespace['folds']) if part in group) for part in range(50)]
        actual=assign_folds(np.arange(50),np.arange(50),n_splits=splits)
        np.testing.assert_array_equal(actual,expected);source_cases+=50
    actual=assign_folds(np.arange(50),np.arange(50))
    assert np.bincount(actual).tolist()==[3]*15+[5]
    assert actual[44]==14 and actual[45]==15 and actual[49]==15
    assert assign_folds([0,2,3,5],[0,0,2,2]).tolist()==[0,0,1,1]
    rejections=0
    for parts,originals,kwargs in [([0,3],[0,0],{}),([0,0],[1,0],{}),
                                  ([0],[1],{}),([50],[0],{}),([0],[0],{'n_splits':51})]:
        try:assign_folds(parts,originals,**kwargs)
        except ValueError:rejections+=1
        else:raise AssertionError('invalid grouping accepted')
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_paths')
    crops=[{'frame':f,'actor':a} for f,a in ((320,0),(20,1),(0,0),(310,1),(10,2),(15,0),(10,0))]
    existing={f"crops/synthetic/{r['frame']}_{r['actor']}.png" for r in crops}
    namespace={'os':SimpleNamespace(path=SimpleNamespace(join=os.path.join,exists=lambda p:p in existing))}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<source-crop-inventory>','exec'),namespace)
    expected=namespace['get_paths'](('synthetic','synthetic'),0,'')
    indices=eligible_crop_positions(crops)
    actual=[(crops[i]['frame'],crops[i]['actor']) for i in indices]
    parsed=[tuple(map(int,Path(row[0]).stem.split('_'))) for row in expected]
    assert actual==parsed==[(0,0),(10,0),(20,1),(310,1)]
    assert not any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='args' and n.attr=='seed' for n in ast.walk(main_fn))
    files=['sciona/dfdc_folds.py','scripts/validate_dfdc_folds.py','docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-fold-validation.v1','result':'passed','source_commit':pins['commit'],
       'checks':{'source_part_assignments':source_cases,'independent_default_fold_sizes':True,
                 'original_altered_fold_agreement':True,'invalid_grouping_rejections':rejections,
                 'source_inventory_order_and_caps':True,'source_cli_seed_unused_confirmed':True},
       'adaptations':['Explicit positional part/original references replace filesystem metadata discovery.',
                      'Restrict fold count1..50, validate direct original references, preserve caller input order.',
                      'No unseeded source global row shuffle here; outer lifecycle must declare its record-order/RNG policy.'],
       'limits':'Synthetic partition/crop positions only. This establishes source part-based folds and pairing checks, not real provenance or full lifecycle/promotion.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_folds.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
