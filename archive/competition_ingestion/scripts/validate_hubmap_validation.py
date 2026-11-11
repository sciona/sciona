"""Full-model validation against original loss/metric accumulation statements."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import signal
import numpy as np
import torch
from sciona.hubmap_network import UNET_SERESNEXT101
from sciona.hubmap_validation import validate_batches
from scripts.validate_hubmap_network import source_model
from scripts.validate_hubmap_losses import reference as loss_reference
from scripts.validate_hubmap_training import exact


def source_validation(root,source_root):
    pins=json.loads((root/'docs/reviews/competition_hubmap_source_pins.json').read_text())
    ns=dict(loss_reference(source_root,pins).__globals__)
    for path in ['src/metrics.py','src/02_train/run.py']:
        assert hashlib.sha256((source_root/path).read_bytes()).hexdigest()==pins['files'][path]
    metrics=[n for n in ast.parse((source_root/'src/metrics.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='dice_sum_2']
    assert len(metrics)==1
    exec(compile(ast.Module(body=metrics,type_ignores=[]),'<pinned-dice>','exec'),ns)
    blocks=[n for n in ast.walk(ast.parse((source_root/'src/02_train/run.py').read_text())) if isinstance(n,ast.With) and any(isinstance(s,ast.AugAssign) and ast.unparse(s.target)=='loss_val' for s in n.body)]
    assert len(blocks)==1
    fn=ast.parse('''def source_validate(model,batches,expected_examples,threshold):
    model.eval()
    config=dict(clfhead=True,deepsupervision=True,dice_threshold=threshold)
    device='cpu'
    criterion=torch.nn.BCEWithLogitsLoss()
    criterion_clf=torch.nn.BCEWithLogitsLoss()
    loss_val=0
    val_score_numer=0
    val_score_denom=0
    for data in batches:
        pass
    return dict(loss=loss_val/expected_examples,dice=float(val_score_numer/val_score_denom),numerator=float(val_score_numer),denominator=float(val_score_denom),examples=expected_examples)
''').body[0]
    loop=next(n for n in fn.body if isinstance(n,ast.For));loop.body=[copy.deepcopy(blocks[0])]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<pinned-validation>','exec'),ns)
    return ns['source_validate']


def validate(root,source_root):
    torch.set_num_threads(1)
    original=source_model(root,source_root);validate_original=source_validation(root,source_root)
    torch.manual_seed(149);candidate=UNET_SERESNEXT101((32,32),True,True,None,load_weights=False)
    torch.manual_seed(149);reference=original((32,32),True,True,None,load_weights=False)
    reports=[]
    for threshold in [.3,.5,.7]:
        torch.manual_seed(151)
        batches=[]
        for n in [2,1]:
            images=torch.randn(n,3,32,32);masks=(torch.rand(n,1,32,32)>.8).float();labels=torch.ones(n)
            if n==2:masks[0].zero_();labels[0]=0
            batches.append((images,masks,labels))
        actual=validate_batches(candidate,batches,expected_examples=3,dice_threshold=threshold)
        expected=validate_original(reference,[dict(img=i,mask=m,label=l) for i,m,l in batches],3,threshold)
        assert actual==expected,(actual,expected)
        exact(candidate.state_dict(),reference.state_dict())
        assert all(p.grad is None for m in [candidate,reference] for p in m.parameters())
        reports.append(dict(threshold=threshold,batch_sizes=[2,1],exact_loss_dice_buffers=True,no_gradients=True))
    paths=['sciona/hubmap_validation.py','sciona/hubmap_losses.py','sciona/hubmap_encoder.py','sciona/hubmap_network.py',
        'scripts/validate_hubmap_validation.py','scripts/validate_hubmap_losses.py','scripts/validate_hubmap_network.py',
        'scripts/validate_hubmap_training.py','tests/test_hubmap_validation.py','docs/reviews/competition_hubmap_source_pins.json']
    return dict(approved=False,synthetic_only=True,cases=reports,
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limitations=['Full initialized network on synthetic float32 CPU inputs; no trained accuracy claim.',
            'Zero global Dice denominator rejects explicitly rather than returning source NaN.',
            'Loss terms accumulate separately as Python floats; unequal final batch included.',
            'Complete epoch/checkpoint execution and CDG publication remain.'])


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_hubmap_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(approved=False,cases=report['cases'])))
