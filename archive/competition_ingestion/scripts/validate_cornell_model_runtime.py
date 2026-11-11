"""Full source model optimizer-state resume and model-only continuation probes."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_model import SourceModel
from sciona.cornell_schedule import learning_rate


def main():
    torch.set_num_threads(2)
    source=SourceModel('/private/tmp/sciona_cornell_source')
    caller_rng=torch.random.get_rng_state().clone()
    model=source.build(seed=1729,synthetic=True)
    assert torch.equal(torch.random.get_rng_state(),caller_rng)
    try:source.build(seed=1)
    except ValueError:pass
    else:raise AssertionError('Unspecified initialization accepted')
    generator=torch.Generator().manual_seed(4)
    waveform=torch.randn(2,1,64000,generator=generator)*.1
    labels=torch.zeros((2,1,264));labels[0,0,0]=1;labels[1,0,1]=1
    lam=torch.tensor([.3,.7]);targets=source.mix(labels,lam)
    criterion=source.loss(augmented=False)
    def optimizer(m,peak=.001):
        return torch.optim.AdamW(m.parameters(),lr=peak,betas=(.9,.999),eps=1e-8,weight_decay=.0001,amsgrad=True)
    def step(m,opt,iteration):
        m.train();opt.zero_grad()
        for group in opt.param_groups:group['lr']=learning_rate(iteration,steps_per_epoch=2,epochs=2,peak=.001)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10+iteration)
            output=m((waveform,lam))
            loss,_=criterion(output['clipwise_output'],dict(all_labels=targets,secondary_labels=torch.zeros_like(targets)))
            loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step();return float(loss.detach())
    opt=optimizer(model);step(model,opt,0)
    checkpoint=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),iteration=1)
    expected_loss=step(model,opt,1)
    resumed=source.build(seed=999,synthetic=True);resumed.load_state_dict(checkpoint['model'],strict=True)
    resumed_opt=optimizer(resumed);resumed_opt.load_state_dict(checkpoint['optimizer'])
    assert step(resumed,resumed_opt,checkpoint['iteration'])==expected_loss
    for key,value in model.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
    continuation=source.build(seed=2,synthetic=True);continuation.load_state_dict(checkpoint['model'],strict=True)
    new_optimizer=optimizer(continuation,peak=.0005)
    assert len(new_optimizer.state)==0 and len(resumed_opt.state)>0
    assert learning_rate(0,steps_per_epoch=2,epochs=2,peak=.0005)==.000005
    for key,value in checkpoint['model'].items():torch.testing.assert_close(value,continuation.state_dict()[key],rtol=0,atol=0)
    report=dict(status='passed',complete_densenet121=True,source_classes=264,
        adamw_amsgrad_resume_exact=True,model_only_continuation_preserves_weights_resets_optimizer=True,
        explicit_initialization_required=True,constructor_preserves_torch_rng=True,
        runtime_sha256={name:hashlib.sha256((ROOT/'sciona'/name).read_bytes()).hexdigest() for name in ['cornell_model.py','cornell_schedule.py']},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Synthetic two-second waveforms, full source topology and two optimizer steps. Complete thirty-second population training, selection, thirteen-member lifecycle and pretrained provenance remain separate.')
    (ROOT/'docs/reviews/competition_cornell_model_runtime_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
