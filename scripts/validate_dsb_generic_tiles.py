"""Independent pinned-source qualification of generic volume tiling operations."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import numpy as np
from sciona.atoms.numerical.spatial import volume_tiles
from scripts.audit_competition_dsb_semantics import checked_source

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source_root):
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Explicit single-thread environment required')
    pins=json.loads((ROOT/'docs/reviews/competition_dsb_source_pins.json').read_text())
    tree=ast.parse(checked_source(source_root,pins,'split_combine.py'))
    tree.body=[node for node in tree.body if isinstance(node,ast.ClassDef)]
    if [n.name for n in tree.body]!=['SplitComb']:raise ValueError('Source class inventory changed')
    changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.AugAssign) and isinstance(node.op,ast.Div):
            if not isinstance(node.target,ast.Name) or node.target.id not in ['side_len','margin']:
                raise ValueError('Unexpected source division')
            node.op=ast.FloorDiv();changes+=1
    if changes!=2:raise ValueError('Source integer-division inventory changed')
    scope=dict(np=np);exec(compile(tree,'<pinned-public-split-combine>','exec'),scope)
    rng=np.random.RandomState(17);cases=0
    for shape in [(1,1,3,5),(2,7,9,11),(1,8,8,8)]:
        for stride in [1,2]:
            for margin in [0,2]:
                volume=rng.normal(size=shape)
                reference=scope['SplitComb'](4,2,stride,margin,170)
                expected,grid=reference.split(volume)
                actual,actual_grid=volume_tiles.split_volume_tiles(volume,4,2,margin)
                np.testing.assert_array_equal(actual,expected)
                assert tuple(grid)==actual_grid
                extent=(4+2*margin)//stride
                predictions=rng.normal(size=(len(actual),extent,extent,extent,3,5))
                expected=reference.combine(predictions,nzhw=grid)
                actual=volume_tiles.combine_volume_tiles(predictions,actual_grid,4,stride,margin)
                np.testing.assert_array_equal(actual,expected)
                assert actual.dtype==expected.dtype==np.float32
                cases+=1
    return dict(passed=True,approved=False,synthetic_only=True,catalog_mutations=0,
        source_commit=pins['commit'],source_hashes={'split_combine.py':pins['files']['split_combine.py']},
        geometry_cases=cases,exact_tiles=True,exact_grid=True,exact_reconstruction=True,native_threads=1,
        provider_sha256=sha(Path(inspect.getfile(volume_tiles))),validator_sha256=sha(Path(__file__)),
        source_extractor_sha256=sha(ROOT/'scripts/audit_competition_dsb_semantics.py'),
        synthetic_test_sha256=sha(ROOT/'tests/test_volume_tiles.py'),
        adaptations=['Two integer augmented divisions use Python 3 floor division.'],
        limitations=['Source parity covers finite tensors with two trailing payload axes; other payload ranks have separate synthetic tests.',
            'Reconstruction is float32 and retains padded-grid extent; input-grid coordinate interpretation remains caller-owned.',
            'No graph integration or catalog approval claimed.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    result=validate(parser.parse_args().source_root)
    (ROOT/'docs/reviews/dsb_generic_tiles_source_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
