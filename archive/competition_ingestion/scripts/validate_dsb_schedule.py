"""Read pinned schedule constants and compare all supported learning-rate epochs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_schedule import CLASSIFIER_PROFILES, DETECTOR_STAGES, DETECTOR_RATES, learning_rate


def source_config(source):
    nodes=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript):
            target=node.targets[0]
            if isinstance(target.value,ast.Name) and target.value.id=='config' and isinstance(target.slice,ast.Constant) and target.slice.value in ['lr_stage','lr','augtype','startepoch']:
                nodes.append(node)
    scope=dict(np=np,config={})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-schedule-constants>','exec'),scope)
    return scope['config']


def validate(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    cases=0
    for variant in [None,3,4]:
        model_file='net_detector_3.py' if variant is None else f'net_classifier_{variant}.py'
        config=source_config(checked_source(source_root,pins,'training/classifier/'+model_file))
        stages=DETECTOR_STAGES if variant is None else CLASSIFIER_PROFILES[variant]['stages']
        rates=DETECTOR_RATES if variant is None else CLASSIFIER_PROFILES[variant]['rates']
        assert tuple(config['lr_stage'])==stages and tuple(config['lr'])==rates
        if variant is not None:
            assert config['startepoch']==20
            assert config['augtype']=={key:CLASSIFIER_PROFILES[variant][key] for key in ['flip','swap','rotate','scale']}
        filename='trainval_detector.py' if variant is None else 'trainval_classifier.py'
        source=checked_source(source_root,pins,'training/classifier/'+filename)
        start=source.index('def get_lr(')
        tree=ast.parse(source[start:source.index('\ndef ',start+1)])
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_lr']
        scope=dict(np=np)
        exec(compile(tree,'<pinned-learning-rate>','exec'),scope)
        for epoch in range(1,stages[-1]+1):
            for override in [None,0.,.003]:
                args=SimpleNamespace(lr=override,lr_stage=np.array(stages),lr_preset=rates,
                                     lr_stage2=np.array(stages),lr_preset2=rates)
                assert learning_rate(epoch,stages,rates,override)==scope['get_lr'](epoch,args)
                cases+=1
    paths=['sciona/dsb_schedule.py','tests/test_dsb_schedule.py','scripts/validate_dsb_schedule.py']
    return dict(approved=False,source_commit=pins['commit'],exact_learning_rate_cases=cases,
        classifier_variant_constants_verified=[3,4],
        limitations=['Main-loop orchestration and complete training sequence require further execution validation.',
                     'Variant 4 enables source rotation without rotating classifier coordinates.',
                     'This does not certify a trained checkpoint or catalog artifact.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_schedule_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
