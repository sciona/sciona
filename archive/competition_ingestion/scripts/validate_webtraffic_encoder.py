"""Synthetic explicit-weight encoder checks against native PyTorch GRU and full-width flow."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from sciona.webtraffic_encoder import encode, predict_without_dropout
from sciona.webtraffic_losses import calc_loss, rnn_activation_loss


def canonical(gru, layers):
    result=[]
    for layer in range(layers):
        wi=getattr(gru,f'weight_ih_l{layer}');wh=getattr(gru,f'weight_hh_l{layer}')
        bi=getattr(gru,f'bias_ih_l{layer}');bh=getattr(gru,f'bias_hh_l{layer}')
        depth=wh.shape[1]
        result.append(dict(gate_kernel=torch.cat([wi[:2*depth].T,wh[:2*depth].T],dim=0),
                           gate_bias=bi[:2*depth]+bh[:2*depth],candidate_input_kernel=wi[2*depth:].T,
                           candidate_hidden_kernel=wh[2*depth:].T,candidate_input_bias=bi[2*depth:],
                           candidate_hidden_bias=bh[2*depth:]))
    return result


def validate(root, source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    for filename in ['model.py','hparams.py']:
        assert hashlib.sha256((source/filename).read_bytes()).hexdigest()==pins['files'][filename]
    torch.set_num_threads(1)
    counts=dict(encoder_comparisons=0,input_gradient_comparisons=0,full_width_flows=0)
    for seed in range(8):
        torch.manual_seed(119+seed)
        for layers in [1,2]:
            gru=torch.nn.GRU(5,7,num_layers=layers).double()
            x=torch.randn(3,11,5,dtype=torch.float64,requires_grad=True)
            a,h=encode(x,canonical(gru,layers));b,hb=gru(x.transpose(0,1))
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-13)
            torch.testing.assert_close(h,hb,rtol=1e-12,atol=1e-13)
            ga=torch.autograd.grad(a.square().sum()+h.square().sum(),x,retain_graph=True)[0]
            gb=torch.autograd.grad(b.square().sum()+hb.square().sum(),x)[0]
            torch.testing.assert_close(ga,gb,rtol=1e-11,atol=1e-12)
            counts['encoder_comparisons']+=1;counts['input_gradient_comparisons']+=1
    # Source s32 sequence lengths and hidden width, using invented numeric features.
    depth=267;torch.manual_seed(812)
    gru=torch.nn.GRU(19,depth)
    with torch.no_grad():
        for p in gru.parameters():p.uniform_(-.05,.05)
    cells=canonical(gru,1)
    decoder={
        'gate_kernel':(torch.randn(19+depth,depth*2)*.02).requires_grad_(),
        'gate_bias':torch.ones(depth*2,requires_grad=True),
        'candidate_kernel':(torch.randn(19+depth,depth)*.02).requires_grad_(),
        'candidate_bias':torch.zeros(depth,requires_grad=True)}
    projection=(torch.randn(depth,1)*.02).requires_grad_();bias=torch.zeros(1,requires_grad=True)
    x=torch.randn(2,283,19,requires_grad=True);y=torch.randn(2,63,18)
    prediction,eo,do,_=predict_without_dropout(x,y,torch.randn(2),torch.ones(2),torch.ones(2),cells,[decoder],projection,bias)
    assert prediction.shape==(2,63) and eo.shape==(283,2,267) and do.shape==(63,2,267)
    assert torch.isfinite(prediction).all()
    true=torch.rand(2,63)*2;true[0,0]=float('nan')
    losses=calc_loss(prediction,true)
    loss=losses[1]+rnn_activation_loss(eo,1e-6/283)+rnn_activation_loss(do,5e-6/63)
    loss.backward()
    for parameter in [x,*gru.parameters(),*decoder.values(),projection,bias]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
    counts['full_width_flows']+=1
    paths=['sciona/webtraffic_encoder.py','scripts/validate_webtraffic_encoder.py',
           'sciona/webtraffic_decoder.py','sciona/webtraffic_losses.py',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                equations_source='https://raw.githubusercontent.com/tensorflow/tensorflow/v1.10.0/tensorflow/contrib/cudnn_rnn/python/ops/cudnn_rnn_ops.py',
                limitations=['Native PyTorch CPU comparison, not original TensorFlow or cuDNN execution.',
                             'Synthetic canonical weights; no source random initialization or checkpoint equivalence.',
                             'Full-width prediction and backward pass omit training dropout and disconnected attention branch.',
                             'Optimizer, EMA, batching and lifecycle validation remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_encoder.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
