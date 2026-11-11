"""Full actual 8/5/2-model input gradients and simultaneous-graph comparison."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_ensemble import ORDER,build_ensemble,gradient
from sciona.adversarial_losses import ensemble_loss


def main():
    torch.set_num_threads(2);checks={};rng=torch.get_rng_state().clone()
    for branch in ('targeted_small','targeted_large','non_targeted'):
        models=build_ensemble(branch,{scope:{'kind':'random','seed':1100+i} for i,scope in enumerate(ORDER[branch])})
        x=torch.linspace(-.9,.9,3*299*299).reshape(1,3,299,299)
        labels=None if branch=='non_targeted' else torch.tensor([17])
        result=gradient(models,x,branch=branch,iteration=0,labels=labels)
        assert result['gradient'].shape==x.shape and torch.count_nonzero(result['gradient'])>0
        assert torch.isfinite(result['gradient']).all()
        if branch=='targeted_small':
            # Independent direct autograd graph, retaining both complete networks.
            z=x.clone().requires_grad_();main=[];aux=[]
            for m in models:
                y,e=m(z);main.append(y);aux.append(e['AuxLogits'])
            loss=ensemble_loss(main,aux,labels,branch=branch)
            expected,=torch.autograd.grad(loss,z)
            assert float(loss.detach())==result['loss']
            torch.testing.assert_close(result['gradient'],expected,rtol=3e-4,atol=1e-12)
            del z,main,aux,y,e,loss,expected
        if branch=='non_targeted':
            supplied=(result['labels']+1)%1001
            repeated=gradient(models,x*.99,branch=branch,iteration=1,labels=supplied)
            assert torch.equal(repeated['labels'],supplied)
            del repeated
        assert torch.equal(rng,torch.get_rng_state())
        checks[branch]={'full_models':len(models),'finite_nonzero_input_gradient':True,'inferred_or_supplied_labels':True}
        del models,result,x
        gc.collect()
    paths=['sciona/adversarial_ensemble.py','scripts/validate_adversarial_ensemble.py',
           'sciona/adversarial_losses.py','sciona/adversarial_state_mapping.py',
           'docs/reviews/competition_adversarial_inception_v3.json','docs/reviews/competition_adversarial_inception_v4.json',
           'docs/reviews/competition_adversarial_inception_resnet.json','docs/reviews/competition_adversarial_resnet.json']
    report={'format':'adversarial-ensemble-gradient-validation.v1','result':'passed','checks':checks,
            'simultaneous_full_two_model_graph_comparison':True,'non_targeted_later_labels_frozen':True,
            'limits':'Full actual randomly initialized source-shaped networks on synthetic single-example CPUfloat32. Sequential VJP versus direct graph checked for two-model branch; no historical TF reduction-order parity, pretrained quality or complete iterative lifecycle claim.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_ensemble.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__':main()
