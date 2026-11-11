"""Source-width synthetic optimizer integration and saved-state continuation checks."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import torch
from sciona.webtraffic_train_step import PARAMETER_NAMES,forward,new_optimizer,train_step


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'model.py').read_bytes()).hexdigest()==pins['files']['model.py']
    torch.set_num_threads(1);torch.manual_seed(981)
    shapes=dict(encoder_wi=(801,19),encoder_wh=(801,267),encoder_bi=(801,),encoder_bh=(801,),
                decoder_gate_kernel=(286,534),decoder_gate_bias=(534,),decoder_candidate_kernel=(286,267),
                decoder_candidate_bias=(267,),projection_kernel=(267,1),projection_bias=(1,))
    params={name:torch.randn(shape)*.02 for name,shape in shapes.items()}
    params['decoder_gate_bias'].fill_(1.)
    optimizer=new_optimizer(params);resumed=None
    counts=dict(model_updates=0,resumed_updates=0,parameter_changes=0,prediction_continuations=0,slot_order_rejections=0)
    for step in range(5):
        batch=dict(x=torch.randn(2,283,19),y=torch.randn(2,63,18),previous=torch.randn(2),mean=torch.ones(2),std=torch.ones(2),
                   truth=torch.rand(2,63)*2,gate_mask=torch.rand(2,267)<.9967589439360334,
                   decoder_masks=[dict(state=torch.rand(63,2,267)<.99,output=torch.rand(63,2,267)<.975)])
        batch['truth'][0,0]=float('nan')
        original={k:v.clone() for k,v in params.items()};old_optimizer=copy.deepcopy(optimizer)
        updated,state,metrics=train_step(params,optimizer,batch)
        assert all(torch.isfinite(v) for v in metrics.values())
        for k in PARAMETER_NAMES:
            torch.testing.assert_close(params[k],original[k],rtol=0,atol=0)
            assert torch.isfinite(updated[k]).all() and not torch.equal(updated[k],params[k])
            counts['parameter_changes']+=1
        for a,b in zip(optimizer['adam']['m'],old_optimizer['adam']['m']):torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert state['adam']['step']==step+1
        # Gate bias gradients reach both source bias variables, so both must update.
        delta_input=updated['encoder_bi'][:534]-params['encoder_bi'][:534]
        delta_hidden=updated['encoder_bh'][:534]-params['encoder_bh'][:534]
        torch.testing.assert_close(delta_input,delta_hidden,rtol=1e-4,atol=1e-8)
        if resumed is not None:
            rp,rs,rm=train_step(resumed['parameters'],resumed['optimizer'],batch)
            for k in PARAMETER_NAMES:torch.testing.assert_close(rp[k],updated[k],rtol=0,atol=0)
            for key in ['m','v']:
                for a,b in zip(rs['adam'][key],state['adam'][key]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            with torch.no_grad():
                a=forward(rp,batch,training=False)['predictions'];b=forward(updated,batch,training=False)['predictions']
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            resumed=dict(parameters=rp,optimizer=rs)
            counts['resumed_updates']+=1;counts['prediction_continuations']+=1
        params,optimizer=updated,state;counts['model_updates']+=1
        if step==1:
            buffer=io.BytesIO();torch.save(dict(parameters=params,optimizer=optimizer),buffer);buffer.seek(0)
            resumed=torch.load(buffer,weights_only=True)
    bad=copy.deepcopy(optimizer);bad['parameter_names']=tuple(reversed(PARAMETER_NAMES))
    try:train_step(params,bad,batch)
    except ValueError:counts['slot_order_rejections']+=1
    else:raise AssertionError('Mismatched slot ordering accepted')
    paths=['sciona/webtraffic_train_step.py','scripts/validate_webtraffic_train_step.py','sciona/webtraffic_training_forward.py',
           'sciona/webtraffic_encoder.py','sciona/webtraffic_decoder.py','sciona/webtraffic_dropout.py',
           'sciona/webtraffic_losses.py','sciona/webtraffic_adam.py','docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Explicit-mask synthetic initialization; no source RNG or TensorFlow/cuDNN runtime parity.',
                             'Separate encoder input/recurrent bias variables preserved during optimizer updates.',
                             'In-memory parameter/Adam continuation only; EMA, scheduler and TF checkpoint codec not validated.',
                             'Source full-width model used with synthetic batch2; full source batching and disconnected branch remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_train_step.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
