"""Replay base population construction against both source training scripts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sciona.wheat_base_population import base_populations


def main(root, output):
    cases = 0
    for file, digest in [('effdet_train.py','d469f63af50abdb8a9e704bb756a569fd5f06383d3a5d09bd36b951220dbec4c'),
                         ('faster_rcnn_fpn_train.py','46086931f010b57fdb31e0f613dfa4fdd4a76461d39c742edd288144dbc5a588')]:
        raw=(root/file).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('historical population source drift')
        loop=next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.For) and ast.unparse(n.iter)=='args.folds')
        nodes=[n for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
               and n.targets[0].id in {'valid_df','train_df','warm_df'}]
        assert len(nodes)==5
        code=compile(ast.Module(body=nodes,type_ignores=[]),'<historical-populations>','exec')
        for seed in range(16):
            for fold in range(5):
                primary=pd.DataFrame(dict(fold=np.tile(np.arange(5),4),isbox=np.arange(20)%3!=0,
                    synthetic_row=np.arange(20)),index=np.arange(20)*7+3)
                auxiliary=pd.DataFrame(dict(fold=[fold]*3,isbox=[True,False,True],synthetic_row=np.arange(20,23)))
                warm_only=pd.DataFrame(dict(fold=[fold]*2,isbox=[False,True],synthetic_row=np.arange(23,25)))
                snapshots=[table.copy(deep=True) for table in (primary,auxiliary,warm_only)]
                rng=np.random.RandomState(seed)
                actual=base_populations(primary,auxiliary,warm_only,fold,numpy_rng=rng)
                np.random.seed(seed)
                namespace=dict(pd=pd,df=primary,wheat2017_df=auxiliary,spike_df=warm_only,fold=fold)
                exec(code,namespace)
                for key,ref_key in [('warm','warm_df'),('training','train_df'),('validation','valid_df')]:
                    pd.testing.assert_frame_equal(actual[key],namespace[ref_key],check_exact=True)
                a,b=rng.get_state(),np.random.get_state()
                assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
                for table,snapshot in zip((primary,auxiliary,warm_only),snapshots):
                    pd.testing.assert_frame_equal(table,snapshot,check_exact=True)
                cases+=1
    files=['sciona/wheat_base_population.py','scripts/validate_wheat_base_population.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,exact_population_cases=cases,
        all_folds_and_both_detectors=True,row_order_and_rng_exact=True,inputs_unchanged=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Synthetic in-memory annotation rows only; image loading, auxiliary-source provenance and image augmentation remain separate.',
                'Source pandas operations execute on installed pandas; historical pandas numerical/runtime parity is not claimed.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_population_cases=cases)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source_root,args.output)
