"""Execute selected original scheduling statements with a synthetic step counter."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from sciona.webtraffic_schedule import training_schedule,retained_checkpoint_steps


def reference_function(source):
    tree=ast.parse((source/'trainer.py').read_text())
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    epoch=next(n for n in ast.walk(train) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    inner=next(n for n in epoch.body if isinstance(n,ast.For))
    cadence=next(n for n in inner.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.BinOp) and isinstance(n.test.left.op,ast.Mod))
    save=next(n for n in cadence.body if isinstance(n,ast.If) and ast.unparse(n.test).startswith('save_from_step and'))
    stop=next(n for n in inner.body if isinstance(n,ast.If) and 'max_steps' in ast.unparse(n.test))
    outer_stop=next(n for n in epoch.body if isinstance(n,ast.If) and ast.unparse(n.test)=='max_steps and step > max_steps')
    # Keep original arithmetic, predicates and loop structure; replace IO/metrics with tracing.
    assignments=[]
    for n in train.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'eval_pct','steps_per_epoch','eval_every_step'} for t in n.targets):assignments.append(copy.deepcopy(n))
    save=copy.deepcopy(save);save.body=ast.parse('save_flag = True').body
    cadence=copy.deepcopy(cadence);cadence.body=[save]
    stop=copy.deepcopy(stop)
    inner=copy.deepcopy(inner);inner.iter=ast.parse('range(steps_per_epoch)',mode='eval').body
    inner.body=ast.parse('step += 1\nsave_flag = False').body+[cadence]+ast.parse('events.append((step, epoch, save_flag))').body+[stop]
    outer_stop=copy.deepcopy(outer_stop);outer_stop.body=[ast.Break()]
    epoch=copy.deepcopy(epoch);epoch.body=[inner,outer_stop]
    module=ast.Module(body=assignments+ast.parse('step=0\nevents=[]').body+[epoch],type_ignores=[])
    return compile(ast.fix_missing_locations(module),'<original-trainer-schedule>','exec')


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'trainer.py').read_bytes()).hexdigest()==pins['files']['trainer.py']
    original=reference_function(source)
    counts=dict(schedules=0,step_events=0,retention_checks=0,rejections=0)
    for pages in [2560,5376,25600,256000]:
        for stop,save in [(11500,10500),(15,8),(20,20),(0,0),(None,None)]:
            namespace=dict(real_train_pages=pages,batch_size=256,max_epoch=15,max_steps=stop,save_from_step=save,
                           trainer=SimpleNamespace(has_active=lambda:True))
            exec(original,namespace)
            actual=list(training_schedule(pages,max_epoch=15,max_steps=stop,save_from_step=save))
            assert actual==namespace['events']
            saved=[s for s,_,flag in actual if flag]
            assert retained_checkpoint_steps(actual)==saved[max(0,len(saved)-10):]
            counts['schedules']+=1;counts['step_events']+=len(actual);counts['retention_checks']+=1
    schedule=list(training_schedule(256000))
    assert schedule[-1][0]==11501
    assert retained_checkpoint_steps(schedule)==list(range(10600,11501,100))
    for kwargs in [dict(n_pages=256),dict(n_pages=-1),dict(n_pages=2560,batch_size=0),dict(n_pages=2560,max_steps=-1)]:
        try:list(training_schedule(**kwargs))
        except ValueError:counts['rejections']+=1
        else:raise AssertionError('Invalid schedule accepted')
    paths=['sciona/webtraffic_schedule.py','scripts/validate_webtraffic_schedule.py','docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                findings=['Source stops strictly after max_steps; default can execute step11501.',
                          'Checkpoint cadence is rounded10percent of floored steps per epoch; retains latest10.',
                          'Shared step increment and model updates are fetched together; fetch order is not an execution dependency.'],
                limitations=['No-evaluation final route with all models active and sufficient input batches only.',
                             'No TensorFlow execution order, iterator exhaustion, early stopping or checkpoint codec claim.',
                             'Synthetic population sizes only; complete lifecycle and source runtime evidence remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_schedule.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
