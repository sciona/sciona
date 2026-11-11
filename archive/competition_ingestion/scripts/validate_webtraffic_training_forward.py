"""Synthetic source-dimension forward/backward and deterministic-mask integration."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
from sciona.webtraffic_training_forward import s32_forward
from sciona.webtraffic_encoder import predict_without_dropout
from scripts.validate_webtraffic_encoder import canonical


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    for name in ['model.py','hparams.py']:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pins['files'][name]
    tree=ast.parse((source/'hparams.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='params_s32' for t in n.targets))
    params={k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
    for k,v in dict(train_window=283,rnn_depth=267,encoder_rnn_layers=1,decoder_rnn_layers=1,use_attn=False,
                    gate_dropout=.9967589439360334,encoder_activation_loss=1e-6,decoder_activation_loss=5e-6,
                    encoder_stability_loss=0.,decoder_stability_loss=0.).items():assert params[k]==v
    assert params['decoder_input_dropout'][0]==1. and params['decoder_state_dropout'][0]==.99
    assert params['decoder_output_dropout'][0]==.975 and not params['decoder_variational_dropout'][0]
    torch.set_num_threads(1)
    counts=dict(training_passes=0,evaluation_parity=0,gradient_tensors=0,replayed_masks=0,rejections=0)
    for seed in range(3):
        torch.manual_seed(seed+691)
        gru=torch.nn.GRU(19,267)
        decoder=dict(gate_kernel=(torch.randn(286,534)*.02).requires_grad_(),gate_bias=torch.ones(534,requires_grad=True),
                     candidate_kernel=(torch.randn(286,267)*.02).requires_grad_(),candidate_bias=torch.zeros(267,requires_grad=True))
        projection=(torch.randn(267,1)*.02).requires_grad_();bias=torch.zeros(1,requires_grad=True)
        x=torch.randn(2,283,19,requires_grad=True);y=torch.randn(2,63,18)
        previous=torch.randn(2);mean=torch.ones(2);std=torch.ones(2)
        truth=torch.rand(2,63)*2;truth[0,0]=float('nan')
        gate=torch.rand(2,267)<params['gate_dropout']
        masks=[dict(state=torch.rand(63,2,267)<.99,output=torch.rand(63,2,267)<.975)]
        args=(x,y,previous,mean,std,truth,canonical(gru,1),[decoder],projection,bias)
        result=s32_forward(*args,gate_mask=gate,decoder_masks=masks)
        assert result['item_count']==126 and torch.isfinite(result['total_loss'])
        result['total_loss'].backward()
        for p in [x,*gru.parameters(),*decoder.values(),projection,bias]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
            counts['gradient_tensors']+=1
        counts['training_passes']+=1
        with torch.no_grad():
            replay=s32_forward(*args,gate_mask=gate,decoder_masks=masks)
            torch.testing.assert_close(result['predictions'],replay['predictions'],rtol=0,atol=0)
            counts['replayed_masks']+=1
            evaluation=s32_forward(*args,training=False)
            plain=predict_without_dropout(x,y,previous,mean,std,args[6],[decoder],projection,bias)[0]
            torch.testing.assert_close(evaluation['predictions'],plain,rtol=0,atol=0)
            assert not torch.equal(evaluation['predictions'],result['predictions'])
            torch.testing.assert_close(result['encoder_penalty'],result['encoder_output'].square().sum()*(.5e-6/283))
            torch.testing.assert_close(result['decoder_penalty'],result['decoder_output'].square().sum()*(2.5e-6/63))
            counts['evaluation_parity']+=1
    for kwargs in [dict(gate_mask=None,decoder_masks=masks),dict(gate_mask=gate,decoder_masks=None)]:
        try:s32_forward(*args,**kwargs)
        except ValueError:counts['rejections']+=1
        else:raise AssertionError('Missing training masks accepted')
    paths=['sciona/webtraffic_training_forward.py','scripts/validate_webtraffic_training_forward.py',
           'sciona/webtraffic_encoder.py','sciona/webtraffic_decoder.py','sciona/webtraffic_dropout.py',
           'sciona/webtraffic_losses.py','scripts/validate_webtraffic_encoder.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                encoder_dropout_reference='https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-850/pdf/cuDNN-API.pdf',
                limitations=['Explicit-mask PyTorch training numerics; no TensorFlow RNG or original runtime parity.',
                             'Source dimensions and dropout probabilities verified from pinned hparams; synthetic parameters only.',
                             'Disconnected attention branch, seeded source initialization, optimizer/EMA and checkpoints remain.',
                             'Training batching and complete lifecycle not yet validated.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_training_forward.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
