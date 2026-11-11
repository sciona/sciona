"""Pinned ensemble expression parity plus independent analytical gradients."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_losses import ensemble_loss,infer_labels


def main():
    torch.set_num_threads(2)
    pins=json.loads((ROOT/'docs/reviews/competition_adversarial_source_pins.json').read_text())
    blocks={}
    for branch,file in [('non_targeted','attack_iter.py'),('targeted','target_attack.py')]:
        source=next(s for s in pins['sources'] if s['branch']==branch)
        p=Path('/private/tmp/sciona_adversarial_source')/branch/file
        assert hashlib.sha256(p.read_bytes()).hexdigest()==next(f['sha256'] for f in source['files'] if f['path']==file)
        for fn in ast.parse(p.read_text()).body:
            if not isinstance(fn,ast.FunctionDef) or not fn.name.startswith('graph'):continue
            statements=[n for n in fn.body if isinstance(n,(ast.Assign,ast.AugAssign))
                        and ast.unparse(n).split(' ')[0] in ('logits','auxlogits','cross_entropy')]
            key=branch if branch=='non_targeted' else 'targeted_'+fn.name.split('_')[1]
            blocks[key]=compile(ast.Module(body=statements,type_ignores=[]),'<source-ensemble-loss>','exec')
    cases=0;gradients=0
    for branch,names,weights in [
            ('non_targeted',['v3','adv_v3','ens3_adv_v3','ens4_adv_v3','v4','res_v2','ensadv_res_v2','resnet'],[1,.25,1,1,1,1,1,1]),
            ('targeted_large',['v3','adv_v3','ens3_adv_v3','ens4_adv_v3','ensadv_res_v2'],[4,1,1,1,4]),
            ('targeted_small',['v3','ensadv_res_v2'],[1,2])]:
        for dtype in (torch.float32,torch.float64):
            torch.manual_seed(921)
            main=[torch.randn(3,5,dtype=dtype,requires_grad=True) for _ in names]
            aux=[torch.randn(3,5,dtype=dtype,requires_grad=True) for _ in (names[:-1] if branch=='non_targeted' else names)]
            labels=torch.tensor([0,2,4]);onehot=torch.nn.functional.one_hot(labels,5).to(dtype)
            namespace={'one_hot':onehot,'one_hot_target_class':onehot,
                       'tf':SimpleNamespace(losses=SimpleNamespace(softmax_cross_entropy=
                           lambda target,logits,*,label_smoothing,weights: -(target*logits.log_softmax(1)).sum(1).mean()*weights))}
            for name,value in zip(names,main):namespace['logits_'+name]=value
            for name,value in zip(names,aux):namespace['end_points_'+name]={'AuxLogits':value}
            exec(blocks[branch],namespace)
            expected=namespace['cross_entropy'];actual=ensemble_loss(main,aux,labels,branch=branch)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
            ga=torch.autograd.grad(actual,main+aux,retain_graph=True)
            ge=torch.autograd.grad(expected,main+aux)
            for a,b in zip(ga,ge):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7);gradients+=1
            # Independent dCE/dlogits=(softmax-target)/batch, scaled by head weight.
            for values,w,factor,observed in [(main,weights,1.,ga[:len(main)]),(aux,weights[:len(aux)],.4,ga[len(main):])]:
                fused=sum(v.detach()*weight for v,weight in zip(values,w))/sum(w)
                residual=(fused.softmax(1)-onehot)/3
                for weight,gradient in zip(w,observed):
                    torch.testing.assert_close(gradient,residual*factor*weight/sum(w),rtol=1e-6,atol=1e-7)
            cases+=1
    # Five weak class0 votes lose to three strong class1 probabilities.
    probabilities=[torch.tensor([[.51,.49]]) for _ in range(5)]+[torch.tensor([[0.,1.]]) for _ in range(3)]
    labels=infer_labels(probabilities,iteration=0)
    assert labels.item()==1
    changed=[torch.tensor([[1.,0.]]) for _ in range(8)]
    assert infer_labels(changed,iteration=1,previous=labels).item()==1
    assert infer_labels([torch.tensor([[.5,.5]]) for _ in range(8)],iteration=0).item()==0
    rejected=0
    for fn in [lambda:infer_labels(probabilities[:7],iteration=0),lambda:infer_labels(changed,iteration=1),
               lambda:ensemble_loss(main,aux[:-1],torch.tensor([0,2,4]),branch='targeted_small'),
               lambda:ensemble_loss(main,aux,torch.tensor([0,2,5]),branch='targeted_small')]:
        try:fn()
        except ValueError:rejected+=1
        else:raise AssertionError('invalid ensemble input accepted')
    paths=['sciona/adversarial_losses.py','scripts/validate_adversarial_losses.py','docs/reviews/competition_adversarial_source_pins.json']
    report={'format':'adversarial-losses-validation.v1','result':'passed',
            'checks':{'source_ensemble_loss_cases':cases,'head_gradients_compared':gradients,
                      'independent_analytical_gradients':True,'probability_consensus_not_majority':True,
                      'frozen_labels_and_first_class_tie':True,'invalid_inputs_rejected':rejected},
            'limits':'Pinned expression AST with explicit Torch softmax-CE shim; synthetic logits. No historical TensorFlow binary or backbone/input-gradient/iterative execution claim.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_losses.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
