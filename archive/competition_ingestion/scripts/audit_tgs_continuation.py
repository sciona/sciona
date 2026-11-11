"""Verify private completed-fit receipts before continuing the unchanged TGS run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def audit(runtime,references):
    runtime=Path(runtime).resolve()
    if runtime==ROOT or ROOT in runtime.parents:
        raise ValueError('Private runtime must be outside the repository')
    state_path=runtime/'state/state.json'
    original=state_path.read_bytes()
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    # The unchanged runner reconstructs its complete source/runtime/reference
    # context and rejects mismatched saved state before any training is launched.
    run=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_tgs_synthetic_workflow.py'),
        '--runtime-directory',str(runtime),'--reference-directory',str(references),
        '--threads','1','--prepare-only'],cwd=ROOT,env=env,capture_output=True,text=True)
    if run.returncode:
        raise ValueError('Unchanged runner rejected continuation context; inspect private diagnostics')
    preparation=json.loads(run.stdout)
    if preparation!=dict(prepared=True,training_executed=False,fit_count=63,epoch_ceiling=6530):
        raise ValueError('Complete reviewed training budget required')
    import torch
    from threadpoolctl import threadpool_limits
    from sciona.tgs_checkpoint_store import CheckpointStore
    from sciona.tgs_workflow_state import WorkflowStateStore
    from sciona.tgs_schedule import next_action
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with (runtime/'run.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if state_path.read_bytes()!=original:
            raise ValueError('State changed during context verification')
        envelope=json.loads(original)
        state=WorkflowStateStore(runtime/'state',context_sha256=envelope['context_sha256']).load()
        plan=json.loads((ROOT/'docs/reviews/competition_tgs_training_plan.json').read_text())
        fits={fit['key']:fit for fit in plan['fits']}
        for key,result in state['fits'].items():
            if (key not in fits or result.get('fit')!=key or result.get('fit_contract_sha256')!=
                    hashlib.sha256(json.dumps(fits[key],sort_keys=True).encode()).hexdigest()):
                raise ValueError('Completed fit differs from reviewed plan')
        receipts={}
        def visit(value):
            if isinstance(value,dict):
                if {'sha256','bytes','tensor_count'}<=set(value):
                    prior=receipts.setdefault(value['sha256'],value)
                    if prior!=value:raise ValueError('Conflicting checkpoint receipts')
                for child in value.values():visit(child)
            elif isinstance(value,(list,tuple)):
                for child in value:visit(child)
        visit(state)
        if state['fits'] and not receipts:raise ValueError('Completed fits require checkpoint receipts')
        store=CheckpointStore(runtime/'checkpoints')
        for receipt in receipts.values():
            tensors=store.get(receipt)
            del tensors
        action=next_action(plan,state['fits'],state['rounds'])
        if state_path.read_bytes()!=original:raise ValueError('Continuation audit mutated state')
    return dict(passed=True,training_executed=False,approved=False,
        completed_fits=len(state['fits']),completed_rounds=len(state['rounds']),
        verified_unique_checkpoints=len(receipts),next_action_kind=action['kind'],
        exact_runner_context_verified=True,private_state_unchanged=True,
        planned_fits=63,epoch_ceiling=6530,resource_policy=dict(torch_threads=1,native_threads=1),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Completed-fit continuation only; an unrecorded in-progress fit restarts from its deterministic seed and qualified predecessor.',
            'Checkpoint weights do not restore interrupted optimizer state. This does not establish full execution or CDG approval.',
            'The previous process is absent; its termination cause was not recovered.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory',required=True,type=Path)
    parser.add_argument('--reference-directory',required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.runtime_directory,args.reference_directory)
    (ROOT/'docs/reviews/tgs_continuation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
