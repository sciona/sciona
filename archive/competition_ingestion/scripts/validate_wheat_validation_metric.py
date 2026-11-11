"""Compare source validation scoring, including integer and threshold boundaries."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from sciona.wheat_validation_metric import fasterrcnn_validation_score


def main(source, output):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'dd9e40d9a5ecdef68eac2055a3026b065e8aab7afeafc106fd2024298279053b':
        raise ValueError('historical scoring source drift')
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)]
    for node in nodes:
        node.decorator_list = []
    namespace = dict(np=np, iou_thresholds=[.5,.55,.6,.65,.7,.75])
    exec(compile(ast.Module(body=nodes,type_ignores=[]), '<historical-scoring>', 'exec'), namespace)
    rng = np.random.default_rng(1327)
    cases = 0
    for seed in range(128):
        records = []
        for count in (0,1,7):
            starts = rng.uniform(0,900,(count,2))
            truth = np.concatenate([starts, starts+rng.uniform(.01,120,(count,2))],axis=1)
            boxes = np.concatenate([truth,truth+.6,np.array([[-3.4,-1.2,37.9,48.1],[1001.,1003.,1077.,1099.]])])
            scores = rng.choice([.49,.5,.500001,.8,1.],size=len(boxes))
            if seed==0:
                boxes = np.empty((0,4));scores=np.empty(0)
            if seed==1:
                truth=np.array([[0.,0.,0.,0.]])
            records.append(dict(pred_boxes=boxes,scores=scores,gt_boxes=truth))
        original = [{k:v.copy() for k,v in row.items()} for row in records]
        prepared = [dict(pred_boxes=row['pred_boxes'].clip(0,1023).astype(int),
                         scores=row['scores'].copy(),gt_boxes=row['gt_boxes'].astype(int)) for row in records]
        expected = namespace['calculate_final_score'](prepared, score_threshold=.5)
        actual = fasterrcnn_validation_score(records)
        assert actual == expected and np.isfinite(actual)
        for row, before in zip(records, original):
            for key in row:
                assert np.array_equal(row[key],before[key])
        cases += 1
    box=np.array([[10.,10.,20.,20.]])
    assert np.isclose(fasterrcnn_validation_score([dict(pred_boxes=box,scores=np.array([1.]),gt_boxes=box)]),1.)
    assert np.isclose(fasterrcnn_validation_score([dict(pred_boxes=np.repeat(box,2,axis=0),scores=np.ones(2),gt_boxes=box)]),.5)
    files=['sciona/wheat_validation_metric.py','scripts/validate_wheat_validation_metric.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=digest,
        exact_batch_score_cases=cases,images_per_batch=3,inclusive_coordinates_and_strict_score_threshold=True,
        prediction_order_preserved=True,input_arrays_unchanged=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Source Python scoring arithmetic; historical Numba compiler equivalence is not claimed.',
                'Synthetic predictions only; full detector training and validation populations remain pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_batch_score_cases=cases)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.output)
