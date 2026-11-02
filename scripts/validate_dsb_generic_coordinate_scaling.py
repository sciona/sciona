"""Direct source coordinate-expression qualification using synthetic rows."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
from sciona.atoms.numerical.spatial import coordinate_scaling as candidate
from scripts.audit_competition_dsb_semantics import checked_source
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source_root):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Explicit single-thread environment required')
    pins=json.loads((ROOT/'docs/reviews/competition_dsb_source_pins.json').read_text())
    name='training/prepare.py'
    tree=ast.parse(checked_source(source_root,pins,name))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='savenpy_luna')
    statements=[n for n in ast.walk(function) if isinstance(n,ast.Assign) and len(n.targets)==1
                and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].value,ast.Name)
                and n.targets[0].value.id=='label2']
    if len(statements)!=3:raise ValueError('Source coordinate assignment inventory changed')
    statements.sort(key=lambda n:n.lineno)
    code=compile(ast.Module(body=statements,type_ignores=[]),'<pinned-coordinate-scaling>','exec')
    cases=0
    for count in (0,1,7):
        for spacing in ([1.,1.,1.],[.7,1.3,2.1]):
            for resolution in ([1.,1.,1.],[.6,1.1,1.7]):
                rows=np.arange(count*4,dtype=float).reshape(count,4)*.37-2
                spacing=np.asarray(spacing);resolution=np.asarray(resolution)
                bounds=np.array([[2,9],[4,11],[1,8]])
                scope=dict(np=np,label2=rows.T.copy(),spacing=spacing,resolution=resolution,extendbox=bounds)
                exec(code,scope)
                actual=candidate.scale_coordinate_rows(rows,np.r_[spacing,spacing[1]],np.r_[resolution,resolution[1]],np.r_[bounds[:,0],0])
                np.testing.assert_array_equal(actual,scope['label2'].T)
                cases+=1
    return dict(passed=True,approved=False,synthetic_only=True,catalog_mutations=0,cases=cases,exact_coordinates=True,
        source_commit=pins['commit'],source_hashes={name:pins['files'][name]},native_threads=1,
        provider_sha256=sha(Path(inspect.getfile(candidate))),validator_sha256=sha(Path(__file__)),
        test_sha256=sha(ROOT/'tests/test_coordinate_scaling.py'),
        limitations=['Float64 ordered scale/divide/subtract arithmetic qualified.',
                    'World displacement, flips, axis ordering and sentinel handling remain recipe policy.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    result=validate(parser.parse_args().source_root)
    (ROOT/'docs/reviews/dsb_generic_coordinate_scaling_source_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
