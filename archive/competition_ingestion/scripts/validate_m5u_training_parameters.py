"""Compare training controls with hash-pinned source literals and expressions."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.m5u_training import PARAMETERS,QUANTILES,QUANTILE_WEIGHTS,schedule


def main(source):
    pins_path=ROOT/'docs/reviews/competition_m5_uncertainty_source_pins.json'
    pins=json.loads(pins_path.read_text());digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest==next(v for k,v in pins['files'].items() if k.endswith('quantiles_sage_1to9_eval.py'))
    tree=ast.parse(source.read_text());values={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('P_DICT','QUANTILE_LEVELS','QUANTILE_WTS','SS_PWR','SS_SS'):
                    values[target.id]=ast.literal_eval(node.value)
    assert PARAMETERS==values['P_DICT']
    assert QUANTILES==tuple(values['QUANTILE_LEVELS']) and QUANTILE_WEIGHTS==tuple(values['QUANTILE_WTS'])
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='runQBags']
    assert len(calls)==1
    kwargs={k.arg:k.value for k in calls[0].keywords}
    iteration=compile(ast.Expression(kwargs['n_iter']),'<pinned-search-count>','eval')
    fraction=compile(ast.Expression(kwargs['data'].args[0]),'<pinned-sample-fraction>','eval')
    comparisons=0
    for level in PARAMETERS:
        for speed,super_speed in ((False,False),(True,False),(False,True)):
            for factor in (.1,.5,1.):
                expected=schedule(level,factor,speed=speed,super_speed=super_speed)
                scope=dict(level=level,SPEED=speed,SUPER_SPEED=super_speed,
                           level_os={level:1/factor},SS_PWR=values['SS_PWR'],
                           SS_FRAC=values['P_DICT'][level][0]*values['SS_SS']/(5 if super_speed else 2 if speed else 1),
                           __builtins__={'int':int})
                assert expected['iterations']==eval(iteration,scope)
                assert math.isclose(expected['fraction'],eval(fraction,scope),rel_tol=1e-14,abs_tol=0.)
                assert expected['scale_range']==values['P_DICT'][level][1]
                comparisons+=1
    paths=[ROOT/'sciona/m5u_training.py',ROOT/'tests/test_m5u_training.py',Path(__file__).resolve(),pins_path]
    report=dict(status='passed',approved=False,catalog_mutations=0,source_software_sha256=digest,
                checks=dict(source_schedule_comparisons=comparisons,parameter_levels=14,quantiles=9,quantile_weights=9),
                sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                scope='Pinned numerical configuration and expressions; explicit independent random streams and aggregate normalization correction remain.')
    (ROOT/'docs/reviews/competition_m5_uncertainty_training_parameters.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    main(parser.parse_args().source)
