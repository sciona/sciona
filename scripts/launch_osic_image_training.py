"""Detach a bounded full-schedule run with durable logs and terminal status.

This launches a fresh run, not checkpoint continuation. Existing runtime and
control directories are never reused. Reserve one shared CPU slot before use.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write_status(control,record):
    temporary=control/'status.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(control/'status.json')


def supervise(dependencies,directory,control):
    command=[sys.executable,'-u',str(ROOT/'scripts/run_osic_synthetic_image_training.py'),
        '--dependencies',str(dependencies),'--runtime-directory',str(directory)]
    record=dict(format='osic-training-supervision.v1',supervisor_pid=os.getpid(),
        state='starting',native_threads=1,started_unix=time.time(),fresh_full_schedule=True)
    write_status(control,record)
    try:
        with (control/'worker.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
            record.update(state='running',runner_pid=process.pid)
            write_status(control,record)
            returncode=process.wait()
        complete=directory/'result.json'
        result=json.loads(complete.read_text()) if complete.is_file() else {}
        qualified=(returncode==0 and result.get('completed') is True and result.get('epochs')==30
            and result.get('optimizer_updates')==960 and result.get('replay',{}).get('passed') is True)
        record.update(state='completed' if qualified else 'failed',returncode=returncode,
            complete_schedule_verified=qualified,ended_unix=time.time())
        write_status(control,record)
        return 0 if qualified else 1
    except BaseException as error:
        record.update(state='supervisor_error',exception_type=type(error).__name__,ended_unix=time.time())
        write_status(control,record)
        raise


def launch(dependencies,directory,control):
    if directory.exists():
        raise ValueError('Fresh runtime directory required; previous runs are preserved')
    control.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                 'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
        env[name]='1'
    with (control/'supervisor.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--supervise',
            '--dependencies',str(dependencies),'--runtime-directory',str(directory),'--control-directory',str(control)],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    (control/'launch.json').write_text(json.dumps(dict(supervisor_pid=process.pid,
        native_threads=1,fresh_full_schedule=True),indent=2)+'\n')
    return dict(supervisor_pid=process.pid,launched=True,native_threads=1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependencies',required=True,type=Path)
    parser.add_argument('--runtime-directory',required=True,type=Path)
    parser.add_argument('--control-directory',required=True,type=Path)
    parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    paths=[p.resolve() for p in (args.dependencies,args.runtime_directory,args.control_directory)]
    if args.supervise:
        sys.exit(supervise(*paths))
    print(json.dumps(launch(*paths)))
