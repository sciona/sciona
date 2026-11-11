"""Synthetic source-batch256 three-model update and EMA inference integration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from sciona.webtraffic_assembly import assemble_features
from sciona.webtraffic_batches import window_batches
from sciona.webtraffic_initialization import uniform_shapes,initialize_from_uniforms
from sciona.webtraffic_lifecycle import batch_inputs,initial_lifecycle,advance,save_checkpoint,predict_checkpoints


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    for name in ['model.py','trainer.py','hparams.py']:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pins['files'][name]
    torch.set_num_threads(2);rng=np.random.RandomState(997);generator=torch.Generator().manual_seed(997)
    pages=sorted(f'SyntheticFullBatch{i}_{j}_{site}_{agent}' for i,site in enumerate(
        ['en.wikipedia.org','fr.wikipedia.org','commons.wikimedia.org','www.mediawiki.org']) for j in range(17)
        for agent in ['all-access_spider','desktop_all-agents','mobile-web_all-agents','all-access_all-agents'])
    values=rng.poisson(np.linspace(4,25,len(pages))[:,None],size=(len(pages),800)).astype(np.float32)
    values[0]=0;values[1,::19]=np.nan
    frame=pd.DataFrame(values,index=np.array(pages,dtype=object),columns=pd.date_range('2000-01-01',periods=800))
    tensors,_=assemble_features(frame,add_days=63)
    records=[]
    for _ in range(3):
        stream=window_batches(tensors,rng.permutation(len(pages)),iter(rng.randint(0,454,size=1000)),
                              data_days=800,batch_size=256,epochs=None,completeness=.01)
        record=next(stream);assert len(record[0])==256;records.append(record)
    parameters=[];batches=[]
    for record in records:
        draws={k:torch.rand(shape,generator=generator) for k,shape in uniform_shapes(record[1].shape[-1],record[5].shape[-1]).items()}
        parameters.append(initialize_from_uniforms(draws,record[1].shape[-1],record[5].shape[-1]))
        gate=torch.rand(256,267,generator=generator)<.9967589439360334
        masks=[dict(state=torch.rand(63,256,267,generator=generator)<.99,output=torch.rand(63,256,267,generator=generator)<.975)]
        batches.append(batch_inputs(record,gate,masks))
    state=initial_lifecycle(parameters)
    # Explicit synthetic post-update observation; not claimed as original TF order.
    observations=[dict(parameter_phase='after',global_step=1) for _ in range(3)]
    started=time.monotonic();updated,metrics=advance(state,batches,observations)
    assert updated['shared_step']==1
    changes=0
    for original,trained,metric in zip(state['models'],updated['models'],metrics):
        assert all(torch.isfinite(v) for v in metric.values())
        assert metric['gradient_norm']>0
        for key,p in trained['parameters'].items():
            assert torch.isfinite(p).all() and not torch.equal(p,original['parameters'][key]);changes+=1
        assert trained['optimizer']['adam']['step']==1
    checkpoints=save_checkpoint([],updated)
    prediction=list(window_batches(tensors,np.arange(len(pages)),[],data_days=800,batch_size=256,epochs=1,
                                   training=False,completeness=.01))
    assert [len(r[0]) for r in prediction]==[256,15]
    output,models=predict_checkpoints(checkpoints,prediction,frame.index)
    assert output.shape==(272,63) and (output.values>=0).all() and output.values.dtype==np.int64
    assert (output.loc[pages[0]]==0).all()
    assert all(m.shape==(271,63) and np.isfinite(m.values).all() for m in models)
    paths=['scripts/validate_webtraffic_full_batch.py']+[str(p.relative_to(root)) for p in sorted((root/'sciona').glob('webtraffic_*.py'))]
    paths+=['docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,
                checks=dict(model_updates=3,batch_size=256,parameter_changes=changes,forecast_rows=272,
                            forecast_days=63,prediction_batches=2,ema_models=3),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['One shared update, not full11501-step training duration.',
                             'Synthetic post-update EMA observation explicitly chosen; original order unresolved.',
                             'No original TensorFlow RNG/runtime or opaque checkpoint codec parity.',
                             'Disconnected source attention branch and promotion-contract validation remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_full_batch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
