"""Compare synthetic proposal annotation to pinned source constructor statements."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.audit_competition_dsb_semantics import checked_source
from sciona.dsb_intake import annotate_classifier_proposals


def validate(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_dsb_source_pins.json').read_text())
    layers=ast.parse(checked_source(source_root,pins,'training/classifier/layers.py'))
    functions=[n for n in layers.body if isinstance(n,ast.FunctionDef) and n.name in ['iou','nms']]
    scope={'np':np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<pinned-iou-nms>','exec'),scope)
    loader=ast.parse(checked_source(source_root,pins,'training/classifier/data_classifier.py'))
    cls=next(n for n in loader.body if isinstance(n,ast.ClassDef) and n.name=='DataBowl3Classifier')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    loop=next(n for n in init.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='idx')
    # Select computation only. The omitted statements load arrays and retain lists.
    statements=[]
    for n in loop.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
            name=n.targets[0].id
            if name=='pbb_label' or (name=='pbb' and not (isinstance(n.value,ast.Call) and
                isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='load')):
                statements.append(n)
        elif isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='p':statements.append(n)
    assert len(statements)==4
    code=compile(ast.Module(body=statements,type_ignores=[]),'<pinned-proposal-annotation>','exec')
    count=0
    for dtype in [np.float32,np.float64]:
        for seed in range(6):
            rng=np.random.RandomState(seed)
            proposals=np.column_stack((rng.uniform(-2,4,12),rng.uniform(0,40,(12,3)),rng.uniform(4,20,12))).astype(dtype)
            proposals[0,0]=-1.;proposals[2,1:]=proposals[1,1:]
            boxes=proposals[1:5,1:].copy() if seed%2 else np.empty((0,4),dtype=dtype)
            for threshold in [0.,.05,1.]:
                state={**scope,'pbb':proposals.copy(),'lbb':boxes,
                       'config':dict(conf_th=-1.,nms_th=.05,detect_th=threshold)}
                exec(code,state)
                actual,known=annotate_classifier_proposals(proposals,boxes,detection_threshold=threshold)
                np.testing.assert_array_equal(actual,state['pbb'])
                np.testing.assert_array_equal(known,state['pbb_label'])
                count+=1
    paths=['sciona/dsb_intake.py','tests/test_dsb_intake.py','scripts/validate_dsb_intake.py']
    return dict(approved=False,synthetic_only=True,source_commit=pins['commit'],exact_annotation_cases=count,
        limitations=['Source array loading omitted; runtime arrays supplied explicitly.',
                     'Validation crop assembly has unit tests but is not covered by this comparison.'],
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_dsb_intake_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
