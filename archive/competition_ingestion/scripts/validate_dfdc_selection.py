"""Pinned submission ordering, explicit plan gates and actual snapshot selection."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_selection import SOURCE_REQUESTS,SOURCE_RUNS,selection_plan,select_snapshots
from sciona.dfdc_checkpoints import finish_epoch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    for name in ('predict_submission.sh','train.sh','configs/b7.json'):
        assert hashlib.sha256((args.source_root/name).read_bytes()).hexdigest()==next(
            p['sha256'] for p in pins['files'] if p['path']==name)
    text=(args.source_root/'predict_submission.sh').read_text()
    requests=tuple(tuple(map(int,row)) for row in re.findall(
        r'final_(\d+)_DeepFakeClassifier_tf_efficientnet_b7_ns_(\d+)_(\d+)',text))
    assert requests==SOURCE_REQUESTS and len(requests)==7
    config=json.loads((args.source_root/'configs/b7.json').read_text())
    seeds=tuple(map(int,re.findall(r'--seed (\d+)',(args.source_root/'train.sh').read_text())))
    assert tuple((seed,0,config['optimizer']['schedule']['epochs']) for seed in seeds)==SOURCE_RUNS
    rejected=0
    def reject(fn):
        nonlocal rejected
        try:fn()
        except ValueError:rejected+=1
        else:raise AssertionError('invalid selection accepted')
    reject(selection_plan)
    # Explicitly supplied extension is accepted and exposed; never a default.
    extended=tuple((seed,fold,41 if seed==888 else count) for seed,fold,count in SOURCE_RUNS)
    plan=selection_plan(extended)
    assert plan['source_requests_preserved'] and not plan['source_runs_preserved']
    assert SOURCE_RUNS[3][2]==40
    snapshots={}
    model=torch.nn.Linear(2,1).eval()
    for position,(seed,fold,epoch) in enumerate(SOURCE_REQUESTS):
        with torch.no_grad():model.bias.fill_(position)
        events=finish_epoch(model,epoch=epoch,best_loss=.8,validation_loss=.4)['events']
        snapshots[(seed,fold,epoch)]=events[1]
    selected=select_snapshots(plan,dict(reversed(list(snapshots.items()))))
    assert [int(s['state_dict']['bias'].item()) for s in selected]==list(range(7))
    assert [s['epoch'] for s in selected]==[37,20,30,32,38,41,24]
    assert all(s['bce_best']==.8 for s in selected)
    assert all(s is snapshots[k] for s,k in zip(selected,SOURCE_REQUESTS))
    reject(lambda:selection_plan(extended,SOURCE_REQUESTS+SOURCE_REQUESTS[:1]))
    reject(lambda:selection_plan(extended+extended[:1]))
    reject(lambda:selection_plan(extended,((111,0,True),)))
    reject(lambda:selection_plan(extended,SOURCE_REQUESTS[:-1]))
    reject(lambda:select_snapshots(dict(plan,source_runs_preserved=True),snapshots))
    for field,value in [('epoch',40),('kind','best_dice'),('state_dict',{}),('bce_best',float('nan'))]:
        broken=copy.deepcopy(snapshots);broken[(888,0,40)][field]=value
        reject(lambda:select_snapshots(plan,broken))
    missing=dict(snapshots);del missing[(888,0,40)]
    reject(lambda:select_snapshots(plan,missing))
    files=['sciona/dfdc_selection.py','sciona/dfdc_checkpoints.py','scripts/validate_dfdc_selection.py',
           'docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-selection-validation.v1','result':'passed','source_commit':pins['commit'],
            'checks':{'source_seven_request_order_exact':True,'source_five_run_defaults_exact':True,
                      'inconsistent_source_default_rejected':True,'explicit_extension_flagged':True,
                      'numbered_snapshot_order_and_prior_best_metadata':True,'invalid_cases_rejected':rejected},
            'limits':'Synthetic linear-model numbered snapshots test selection only. Explicit41-epoch plan is a caller adaptation, not evidence the original seed888 trained41 epochs. No historical weights or full five-run training tested.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
