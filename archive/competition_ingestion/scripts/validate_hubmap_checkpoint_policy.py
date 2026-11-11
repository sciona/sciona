"""Compare source checkpoint/control-flow decisions without writing model files."""
import argparse
import ast
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from sciona.hubmap_checkpoint_policy import CheckpointPolicy


class RemovePrints(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print':return ast.Pass()
        return self.generic_visit(node)


def reference(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    path=source_root/'src/02_train/run.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['src/02_train/run.py']
    tree=ast.parse(path.read_text());blocks=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.For) and ast.unparse(node.target)=='epoch':
            start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.If) and ast.unparse(n.test)=="config['early_stopping']")
            blocks=node.body[start:start+4]
    assert len(blocks)==4 and all(isinstance(n,ast.If) for n in blocks)
    fn=ast.parse('''def decide(records,config):
    events=[]
    model=SimpleNamespace(state_dict=lambda:{})
    output_path=''
    seed=0
    fold=0
    def save(state,path):
        role='best_loss' if 'bestloss' in path else ('best_score' if 'bestscore' in path else 'snapshot')
        events[-1]['save'].append(role)
    def step(*args):
        events[-1]['step_scheduler']=True
    torch=SimpleNamespace(save=save)
    scheduler=SimpleNamespace(step=step)
    loss_val_best=1e99
    val_score_best=-1e99
    val_score_best2=-1e99
    epoch_best=0
    counter_ES=0
    for epoch,loss_val,val_score in records:
        events.append(dict(save=[],stop=True,step_scheduler=False))
        pass
        events[-1]['stop']=False
    return events,dict(best_loss=loss_val_best,score_at_best_loss=val_score_best,best_score=val_score_best2,best_epoch=epoch_best,stale_epochs=counter_ES)
''').body[0]
    loop=next(n for n in fn.body if isinstance(n,ast.For))
    loop.body[1:2]=[RemovePrints().visit(copy.deepcopy(n)) for n in blocks]
    ns=dict(SimpleNamespace=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<pinned-checkpoint-control>','exec'),ns)
    return ns['decide']


def validate(root,source_root):
    source=reference(root,source_root);cases=0;decisions=0
    trajectories=[[(i,10.-i*.1,i*.01) for i in range(1,41)],
                  [(i,1.,i*.01) for i in range(1,41)],
                  [(i,1.+(i%3)*.1,.5+(i%5)*.01) for i in range(1,41)]]
    for records in trajectories:
        for enabled in [False,True]:
            for patience in [0,1,3,50]:
                for period in [3,19]:
                    config=dict(early_stopping=enabled,patience=patience,lr_scheduler_name='CosineAnnealingLR',lr_scheduler={'CosineAnnealingLR':dict(t0=period)})
                    expected,state=source(records,config)
                    policy=CheckpointPolicy();actual=[]
                    for epoch,loss,score in records:
                        event=policy.decide(epoch=epoch,validation_loss=loss,validation_score=score,early_stopping=enabled,patience=patience,snapshot_period=period)
                        actual.append(event)
                        if event['stop']:break
                    assert actual==expected
                    assert asdict(policy)==state
                    cases+=1;decisions+=len(actual)
    paths=['sciona/hubmap_checkpoint_policy.py','scripts/validate_hubmap_checkpoint_policy.py','tests/test_hubmap_validation.py',
           'docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False,synthetic_only=True,exact_trajectories=cases,exact_epoch_decisions=decisions,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Only scalar checkpoint/early-stop/scheduler-order decisions; model serialization is intercepted.',
            'Cosine scheduler configuration only; snapshot divisors use fixed configured period.',
            'Complete training graph and catalog approval remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_checkpoint_policy.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
