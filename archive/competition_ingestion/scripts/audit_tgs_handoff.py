"""Audit persisted TGS dependency closure and early-round pseudo-label ledgers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sciona.tgs_pseudo_selection import select_pseudo_labels
from sciona.tgs_schedule import next_action
from sciona.tgs_workflow_state import WorkflowStateStore


def audit_pseudo_round(stage,result):
    if stage not in (1,2) or result.get('stage')!=stage:
        raise ValueError('early pseudo-round stage mismatch')
    masks=np.asarray(result['masks'])
    if masks.dtype!=np.bool_ or masks.ndim!=3 or not len(masks) or masks.shape[1:]!=(101,101):
        raise ValueError('invalid persisted pseudo-masks')
    areas=masks.sum((1,2),dtype=np.int64)
    if not np.array_equal(areas,result['area']):
        raise ValueError('persisted mask areas differ from masks')
    expected=select_pseudo_labels(result['confidence'],areas,result['nonconstant'])
    for key in ['keras_indices','metadata_order']:
        if not np.array_equal(expected[key],result[key]):
            raise ValueError('persisted pseudo-selection differs: '+key)
    if len(result['torch_folds'])!=5 or any(not np.array_equal(a,b) for a,b in zip(expected['torch_folds'],result['torch_folds'])):
        raise ValueError('persisted PyTorch pseudo-folds differ')
    return dict(stage=stage,query_count=len(masks),keras_selected=len(expected['keras_indices']),
                torch_selected_per_fold=[len(fold) for fold in expected['torch_folds']])


def main(runtime,output):
    manifest=json.loads((runtime/'state/state.json').read_text())
    state=WorkflowStateStore(runtime/'state',context_sha256=manifest['context_sha256']).load()
    plan=json.loads(Path('docs/reviews/competition_tgs_training_plan.json').read_text())
    action=next_action(plan,state['fits'],state['rounds'])
    rounds=[audit_pseudo_round(stage,value) for stage,value in state['rounds'].items() if stage in (1,2)]
    pending=[dict(stage=e['stage'],missing_fit_dependencies=len(set(e['requires_fits'])-set(state['fits']))) for e in plan['ensembles']]
    report=dict(passed=True,approved=False,catalog_mutations=0,completed_fits=len(state['fits']),
        completed_rounds=len(state['rounds']),next_action=action,round_dependencies=pending,pseudo_rounds=rounds,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Persisted dependency and pseudo-selection consistency only; confidence values are not recomputed from model predictions.',
                'Final mosaic round requires separate output/provenance validation.',
                'Checkpoint integrity and fit histories are audited separately; full publication remains pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.runtime_directory,args.output)
