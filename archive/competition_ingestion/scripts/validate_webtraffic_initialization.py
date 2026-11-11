"""Initializer contract and full-model synthetic update checks; explicit draws only."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from sciona.webtraffic_initialization import uniform_shapes,initialize_from_uniforms
from sciona.webtraffic_train_step import new_optimizer,train_step


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'model.py').read_bytes()).hexdigest()==pins['files']['model.py']
    init_pin=json.loads((root/'docs/reviews/competition_webtraffic_initializer_source.json').read_text())
    assert hashlib.sha256((source/'tf110_cudnn_layers.py').read_bytes()).hexdigest()==init_pin['sha256']
    tree=ast.parse((source/'model.py').read_text())
    init=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='default_init')
    call=next(n for n in ast.walk(init) if isinstance(n,ast.Call))
    kw={k.arg:ast.literal_eval(k.value) for k in call.keywords if k.arg!='seed'}
    assert kw==dict(factor=1.,mode='FAN_AVG',uniform=True)
    enc=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_encoder')
    random_call=next(n for n in ast.walk(enc) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tf.initializers.random_uniform')
    bounds={k.arg:ast.literal_eval(k.value) for k in random_call.keywords if k.arg!='seed'}
    assert bounds==dict(minval=-.05,maxval=.05)
    counts=dict(initializations=0,kernel_checks=0,bias_checks=0,training_updates=0,rejections=0)
    torch.set_num_threads(1)
    for seed in range(3):
        generator=torch.Generator().manual_seed(seed+982)
        shapes=uniform_shapes(19,18)
        draws={name:torch.rand(shape,generator=generator) for name,shape in shapes.items()}
        # Explicit same-shape draw replay is supported, without claiming TF seed parity.
        draws['encoder_1']=draws['encoder_0'].clone();draws['encoder_2']=draws['encoder_0'].clone()
        params=initialize_from_uniforms(draws,19,18)
        for name in ['decoder_gate_kernel','decoder_candidate_kernel','projection_kernel']:
            limit=math.sqrt(6/sum(shapes[name]))
            expected=draws[name].numpy().astype(np.float64)*(2*limit)-limit
            np.testing.assert_allclose(params[name],expected,atol=2e-8,rtol=2e-5)
            assert params[name].abs().max()<=limit;counts['kernel_checks']+=1
        for i in range(6):
            block=params['encoder_wi' if i<3 else 'encoder_wh'][(i%3)*267:(i%3+1)*267]
            np.testing.assert_allclose(block,draws[f'encoder_{i}'].numpy().astype(np.float64)*.1-.05,atol=5e-9,rtol=2e-5)
            counts['kernel_checks']+=1
        for name,expected in [('encoder_bi',0),('encoder_bh',0),('decoder_gate_bias',1),('decoder_candidate_bias',0),('projection_bias',0)]:
            assert (params[name]==expected).all();counts['bias_checks']+=1
        assert torch.equal(params['encoder_wi'][:267],params['encoder_wi'][267:534])
        batch=dict(x=torch.randn(2,283,19,generator=generator),y=torch.randn(2,63,18,generator=generator),
                   previous=torch.zeros(2),mean=torch.ones(2),std=torch.ones(2),truth=torch.ones(2,63),
                   gate_mask=torch.rand(2,267,generator=generator)<.9967589439360334,
                   decoder_masks=[dict(state=torch.rand(63,2,267,generator=generator)<.99,
                                       output=torch.rand(63,2,267,generator=generator)<.975)])
        updated,state,metrics=train_step(params,new_optimizer(params),batch)
        assert state['adam']['step']==1 and torch.isfinite(metrics['total_loss'])
        assert all(torch.isfinite(v).all() for v in updated.values())
        counts['training_updates']+=1;counts['initializations']+=1
    for bad in ['missing','one','nan','shape']:
        invalid=dict(draws)
        if bad=='missing':invalid.pop('projection_kernel')
        elif bad=='shape':invalid['projection_kernel']=torch.zeros(2)
        else:invalid['projection_kernel']=torch.full((267,1),1. if bad=='one' else float('nan'))
        try:initialize_from_uniforms(invalid,19,18)
        except ValueError:counts['rejections']+=1
        else:raise AssertionError('Invalid initializer draws accepted')
    paths=['sciona/webtraffic_initialization.py','scripts/validate_webtraffic_initialization.py',
           'sciona/webtraffic_train_step.py','sciona/webtraffic_training_forward.py','sciona/webtraffic_encoder.py',
           'sciona/webtraffic_decoder.py','sciona/webtraffic_dropout.py','sciona/webtraffic_losses.py','sciona/webtraffic_adam.py',
           'docs/reviews/competition_webtraffic_initializer_source.json','docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Source distributions/constants, not TensorFlow random sequence or seeded op correlation parity.',
                             'Explicit unit-uniform draws preserve separate six-block encoder layout; no opaque checkpoint codec.',
                             'Disconnected attention initialization and original runtime scheduling remain outside this evidence.',
                             'Full lifecycle/promotion validation remains.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_initialization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
