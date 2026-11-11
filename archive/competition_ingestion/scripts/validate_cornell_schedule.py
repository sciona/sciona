"""Compare Cornell schedule against the exact historical Ignite schedulers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_schedule import learning_rate


def main():
    path=Path('/private/tmp/sciona_cornell_dependencies/pytorch_ignite/ignite/contrib/handlers/param_scheduler.py')
    deps=json.loads((ROOT/'docs/reviews/competition_cornell_historical_dependencies.json').read_text())
    pins=next(p['pins'] for p in deps if p['wheel'].startswith('pytorch_ignite-'))
    digest=next(p['sha256'] for p in pins if p['software_path']=='ignite/contrib/handlers/param_scheduler.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    spec=importlib.util.spec_from_file_location('cornell_historical_scheduler',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reports=[]
    for steps in (1,3,7):
        for epochs in (1,2,5):
            if steps*epochs<2:continue
            for peak in (.001,.0005):
                parameter=torch.nn.Parameter(torch.zeros(1))
                optimizer=torch.optim.AdamW([parameter],lr=peak)
                scheduler=module.ConcatScheduler([
                    module.LinearCyclicalScheduler(optimizer,'lr',peak*.01,peak,steps*2),
                    module.CosineAnnealingScheduler(optimizer,'lr',peak,peak*.01,epochs*steps)],durations=[steps])
                errors=[]
                for iteration in range(steps*epochs):
                    scheduler(None)
                    expected=optimizer.param_groups[0]['lr']
                    actual=learning_rate(iteration,steps_per_epoch=steps,epochs=epochs,peak=peak)
                    assert abs(expected-actual)<1e-15
                    errors.append(abs(expected-actual))
                reports.append(dict(steps_per_epoch=steps,epochs=epochs,peak=peak,max_abs_error=max(errors)))
    failures=0
    for values in [(True,3,2,.001),(-1,3,2,.001),(6,3,2,.001),(0,0,2,.001),(0,3,0,.001),(0,3,2,float('nan')),(0,1,1,.001)]:
        i,s,e,p=values
        try:learning_rate(i,steps_per_epoch=s,epochs=e,peak=p)
        except ValueError:failures+=1
        else:raise AssertionError('Invalid schedule accepted')
    report=dict(status='passed',cases=reports,invalid_schedules_rejected=failures,
        historical_scheduler_sha256=digest,
        runtime_sha256=hashlib.sha256((ROOT/'sciona/cornell_schedule.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Every iteration compared across16source schedules, including epoch transition and truncated cosine horizon. Model-only continuation restarts at zero; full optimizer/checkpoint integration remains separate.')
    (ROOT/'docs/reviews/competition_cornell_schedule_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',schedules=len(reports),invalid_schedules_rejected=failures)))


if __name__=='__main__':main()
