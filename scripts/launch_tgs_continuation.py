"""Supervise an unchanged one-core TGS continuation with private exit diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    runtime=args.runtime_directory.resolve()
    control=args.control_directory.resolve()
    for path in (runtime,control):
        if path==ROOT or ROOT in path.parents:
            raise ValueError('Runtime and diagnostics must remain outside the repository')
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    if args.supervise:
        command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/run_tgs_synthetic_workflow.py'),
            '--runtime-directory',str(runtime),'--reference-directory',str(args.reference_directory.resolve()),
            '--threads','1']
        with (control/'worker.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT)
            (control/'worker.json').write_text(json.dumps(dict(pid=process.pid,threads=1))+'\n')
            code=process.wait()
        (control/'exit.json').write_text(json.dumps(dict(returncode=code,
            workflow_completion_requires_log_and_state_verification=True))+'\n')
    else:
        if not (runtime/'state/state.json').is_file():
            raise ValueError('Existing private workflow state required')
        control.mkdir(parents=True,exist_ok=False)
        command=[str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--supervise',
            '--runtime-directory',str(runtime),'--reference-directory',str(args.reference_directory.resolve()),
            '--control-directory',str(control)]
        with (control/'supervisor.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (control/'supervisor.json').write_text(json.dumps(dict(pid=process.pid))+'\n')
        print(json.dumps(dict(supervisor_pid=process.pid,threads=1,launched=True,completed=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory',type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path,required=True)
    parser.add_argument('--control-directory',type=Path,required=True)
    parser.add_argument('--supervise',action='store_true')
    main(parser.parse_args())
