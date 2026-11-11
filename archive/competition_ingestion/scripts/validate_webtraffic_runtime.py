"""Complete seed-driven synthetic Community runtime and negative boundary checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sciona.webtraffic_runtime import RuntimeConfig,prepare_runtime,train_runtime,forecast_runtime,execute_runtime


def validate(root,source):
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    for name in ['model.py','trainer.py']:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pins['files'][name]
    torch.set_num_threads(1);rng=np.random.RandomState(1171)
    pages=sorted(f'SyntheticRuntime{i}_{site}_{agent}' for i,site in enumerate(
        ['en.wikipedia.org','fr.wikipedia.org','commons.wikimedia.org','www.mediawiki.org'])
        for agent in ['all-access_spider','desktop_all-agents','mobile-web_all-agents','all-access_all-agents'])
    values=rng.poisson(np.arange(4,20)[:,None],size=(16,800)).astype(np.float32);values[0]=0
    frame=pd.DataFrame(values,index=np.array(pages,dtype=object),columns=pd.date_range('2000-01-01',periods=800))
    original=frame.copy(deep=True)
    cfg=RuntimeConfig(batch_size=2,max_steps=1,save_from_step=1,prediction_batch_size=7)
    counts=dict(complete_runs=0,deterministic_replays=0,phase_scenarios=0,rejections=0)
    prepared=prepare_runtime(frame,cfg);trained=train_runtime(prepared);result=forecast_runtime(prepared,trained)
    assert trained['state']['shared_step']==2 and len(trained['checkpoints'])==2
    assert result.shape==(16,63) and result.values.dtype==np.int64
    pd.testing.assert_index_equal(result.columns,pd.date_range(frame.columns[-1]+pd.Timedelta(days=1),periods=63))
    assert (result.loc[pages[0]]==0).all();counts['complete_runs']+=1
    replay=execute_runtime(frame,cfg);pd.testing.assert_frame_equal(result,replay)
    counts['complete_runs']+=1;counts['deterministic_replays']+=1
    alternate=RuntimeConfig(batch_size=2,max_steps=1,save_from_step=1,prediction_batch_size=7,
                            ema_parameter_phase='before',ema_step_phase='before')
    other_prepared=prepare_runtime(frame,alternate);other=train_runtime(other_prepared)
    # Adam parameters remain identical; only declared EMA observations differ.
    difference=False
    for a,b in zip(trained['state']['models'],other['state']['models']):
        for name in a['parameters']:
            torch.testing.assert_close(a['parameters'][name],b['parameters'][name],rtol=0,atol=0)
            difference |= not torch.equal(a['ema'][name],b['ema'][name])
    assert difference;counts['phase_scenarios']+=1
    pd.testing.assert_frame_equal(frame,original)
    for bad_frame,bad_cfg in [(frame.iloc[:,:400],cfg),(frame*0,cfg),
                              (frame,RuntimeConfig(batch_size=2,max_steps=1,save_from_step=100)),
                              (frame,RuntimeConfig(seed=-1)),(frame,RuntimeConfig(ema_step_phase='implicit'))]:
        try:prepare_runtime(bad_frame,bad_cfg)
        except ValueError:counts['rejections']+=1
        else:raise AssertionError('Invalid runtime boundary accepted')
    paths=['scripts/validate_webtraffic_runtime.py']+[str(p.relative_to(root)) for p in sorted((root/'sciona').glob('webtraffic_*.py'))]
    paths+=['docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                declared_semantics={'rng':'CPU NumPy RandomState per-model permutation/offset stream; CPU torch.Generator per-model initialization/dropout stream',
                                    'ema':'Config explicitly selects parameter and shared-step before/after observations; default after/after',
                                    'checkpoint':'Independent in-memory named parameter/Adam/EMA snapshots; latest ten retained',
                                    'forecast':'All63days following runtime data end; no competition-specific file export/date slice'},
                limitations=['Derived Community numerical runtime, not seeded TensorFlow or original race replay.',
                             'Active s32 prediction/training path retained; unconsumed attention branch and TF checkpoint format excluded.',
                             'Full-duration schedule supported but not executed in this short synthetic validation.',
                             'Provider graph, automated semantic review, publication and served execution still required.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
