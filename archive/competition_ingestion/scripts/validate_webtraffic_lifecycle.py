"""Synthetic population -> three-model updates -> saved EMA prediction integration."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sciona.webtraffic_assembly import assemble_features
from sciona.webtraffic_batches import window_batches
from sciona.webtraffic_initialization import uniform_shapes,initialize_from_uniforms
from sciona.webtraffic_lifecycle import batch_inputs,initial_lifecycle,advance,save_checkpoint,predict_checkpoints
from sciona.webtraffic_schedule import training_schedule


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    for name in ['model.py','trainer.py','input_pipe.py']:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pins['files'][name]
    torch.set_num_threads(1);rng=np.random.RandomState(892);generator=torch.Generator().manual_seed(792)
    pages=sorted(f'SyntheticLifecycle{i}_{site}_{agent}' for i,site in enumerate(
        ['en.wikipedia.org','fr.wikipedia.org','commons.wikimedia.org','www.mediawiki.org'])
        for agent in ['all-access_spider','desktop_all-agents','mobile-web_all-agents','all-access_all-agents'])
    values=rng.poisson(np.arange(4,20)[:,None],size=(16,800)).astype(np.float32)
    values[0]=0;values[1,:15]=np.nan
    frame=pd.DataFrame(values,index=np.array(pages,dtype=object),columns=pd.date_range('2000-01-01',periods=800))
    tensors,plain=assemble_features(frame,add_days=63)
    streams=[iter(window_batches(tensors,rng.permutation(16),iter(rng.randint(0,454,size=1000)),data_days=800,
                                 batch_size=2,epochs=None,completeness=.01)) for _ in range(3)]
    first=[next(s) for s in streams]
    xdepth=first[0][1].shape[-1];ydepth=first[0][5].shape[-1]
    parameters=[]
    for _ in range(3):
        draws={k:torch.rand(shape,generator=generator) for k,shape in uniform_shapes(xdepth,ydepth).items()}
        parameters.append(initialize_from_uniforms(draws,xdepth,ydepth))
    state=initial_lifecycle(parameters);checkpoints=[];resumed=None
    counts=dict(shared_updates=0,model_updates=0,resumed_models=0,ema_prediction_models=0,retained_snapshots=0)
    for step,epoch,save in training_schedule(16,batch_size=2,max_steps=2,save_from_step=1):
        records=first if step==1 else [next(s) for s in streams]
        batches=[]
        for record in records:
            n=len(record[0])
            gate=torch.rand(n,267,generator=generator)<.9967589439360334
            masks=[dict(state=torch.rand(63,n,267,generator=generator)<.99,output=torch.rand(63,n,267,generator=generator)<.975)]
            batches.append(batch_inputs(record,gate,masks))
        # Deliberately declared scenario, not a claim about original TF scheduling.
        observations=[dict(parameter_phase='after',global_step=step),dict(parameter_phase='before',global_step=step-1),
                      dict(parameter_phase='after',global_step=step-1)]
        state,metrics=advance(state,batches,observations)
        assert state['shared_step']==step and all(torch.isfinite(m['total_loss']) for m in metrics)
        counts['shared_updates']+=1;counts['model_updates']+=3
        if resumed is not None:
            resumed,_=advance(resumed,batches,observations)
            for a,b in zip(state['models'],resumed['models']):
                for key in ['parameters','ema']:
                    for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
                for key in ['m','v']:
                    for av,bv in zip(a['optimizer']['adam'][key],b['optimizer']['adam'][key]):torch.testing.assert_close(av,bv,rtol=0,atol=0)
                counts['resumed_models']+=1
        if save:checkpoints=save_checkpoint(checkpoints,state)
        if step==1:
            buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0);resumed=torch.load(buffer,weights_only=True)
    prediction=list(window_batches(tensors,np.arange(16),[],data_days=800,batch_size=5,epochs=1,training=False,completeness=.01))
    result,models=predict_checkpoints(checkpoints,prediction,frame.index)
    assert result.shape==(16,63) and all(dtype==np.int64 for dtype in result.dtypes)
    assert (result.loc[pages[0]]==0).all() and (result.values>=0).all()
    assert len(models)==3 and all(m.shape==(15,63) and np.isfinite(m.values).all() for m in models)
    counts['ema_prediction_models']=len(models)
    # Replay serialized checkpoints yields the same final predictions.
    buffer=io.BytesIO();torch.save(checkpoints,buffer);buffer.seek(0)
    restored=torch.load(buffer,weights_only=True)
    replay,_=predict_checkpoints(restored,prediction,frame.index)
    pd.testing.assert_frame_equal(result,replay)
    retained=[]
    for step in range(12):retained=save_checkpoint(retained,dict(shared_step=step,models=[]))
    assert [cp['shared_step'] for cp in retained]==list(range(2,12));counts['retained_snapshots']=len(retained)
    files=['sciona/webtraffic_lifecycle.py','scripts/validate_webtraffic_lifecycle.py']
    files += [str(p) for p in sorted((root/'sciona').glob('webtraffic_*.py')) if str(p.relative_to(root)) not in files]
    files=[str(Path(p).relative_to(root)) if Path(p).is_absolute() else p for p in files]
    files+=['docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
                limitations=['Synthetic three-step/batch2 integration; source width267 and283-to63windows retained.',
                             'Declared mixed EMA observation scenario does not establish original scheduling.',
                             'Explicit synthetic draws/masks/offsets; no TF RNG, checkpoint codec or original runtime parity.',
                             'Full training duration/batch256, disconnected attention and source execution validation remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_lifecycle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
