"""Check group partition parity over previously validated raw tile preparation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.hubmap_tiling import RawTiles
from sciona.hubmap_preparation import prepare_tile
from sciona.hubmap_raw_training import prepare_fold


def validate(root, source):
    pins = json.loads((root/'docs/reviews/competition_hubmap_partition_pins.json').read_text())
    name = 'src/get_fold_idxs_list.py'
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == pins['files'][name]
    nodes = [n for n in ast.parse((source/name).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == 'get_fold_idxs_list']
    assert len(nodes) == 1
    ns = dict(np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]), '<pinned-fold-selection>', 'exec'),ns)
    cases = 0
    for size in [16,32]:
        slides = []
        groups = ['synthetic-a','synthetic-b','synthetic-a','synthetic-c']
        for i in range(4):
            shape = (size+3,size+5)
            image = np.random.RandomState(251+i).randint(0,256,(*shape,3)).astype(np.uint8)
            mask = np.zeros(shape,dtype=np.uint8)
            mask[::i+2] = 1
            slides.append((image,mask))
        # Build the complete population first, then execute the original fold
        # selection. Numerical tile preparation has separate source evidence.
        records = []
        for shift in [0,size//2]:
            for (image,mask),group in zip(slides,groups):
                tiles = RawTiles(image,mask,tile_size=size,shift_h=shift,shift_w=shift)
                order = np.argsort([tiles[i][1].sum() for i in range(len(tiles))])[::-1]
                for i in order:
                    record = prepare_tile(*tiles[i])
                    if record['std_img'] > 10:
                        records.append(dict(**record,patient_number=group))
        frame = pd.DataFrame(records)
        held_out = [['synthetic-a'],['synthetic-b'],['synthetic-c'],
                    ['synthetic-a','synthetic-c'],['synthetic-b','synthetic-b'],['synthetic-c','synthetic-absent']]
        train_indices, valid_indices = ns['get_fold_idxs_list'](frame,held_out)
        for hold, ti, vi in zip(held_out,train_indices,valid_indices):
            train, valid = prepare_fold(slides,groups,hold,tile_size=size)
            for actual,indices in [(train,ti),(valid,vi)]:
                assert actual['rles'] == [records[i]['rle'] for i in indices]
                assert len(actual['images_bgr']) == len(indices)
                for image,i in zip(actual['images_bgr'],indices):
                    np.testing.assert_array_equal(image,records[i]['image_bgr'])
            expected_bins = np.round(frame.iloc[ti]['ratio_masked_area'].to_numpy()*20).astype(int)
            np.testing.assert_array_equal(train['bins'],expected_bins)
            np.testing.assert_array_equal(train['present'],expected_bins>0)
            assert set(frame.iloc[ti]['patient_number']).isdisjoint(frame.iloc[vi]['patient_number'])
            cases += 1
        for invalid_groups in [[],groups]:
            try: prepare_fold(slides,groups,invalid_groups,tile_size=size)
            except ValueError: pass
            else: raise AssertionError('Empty partition accepted')
    paths = ['sciona/hubmap_raw_training.py','sciona/hubmap_raw_population.py','sciona/hubmap_preparation.py',
             'sciona/hubmap_tiling.py','scripts/validate_hubmap_partition.py',
             'docs/reviews/competition_hubmap_partition_pins.json']
    return dict(approved=False,synthetic_only=True,exact_partition_cases=cases,empty_partition_rejections=4,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Partition selection independently executes pinned source; raw numerical preparation covered by separate source comparisons.',
                             'Synthetic runtime group labels only; original dataset group identities neither embedded nor required.',
                             'Raw-to-training full-network comparison and pseudo-label/inference publication gates remain.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_partition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
