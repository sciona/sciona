"""Integrated full-length synthetic Cornell training and continuation."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_model import SourceModel
from sciona.cornell_augmentation import Augmentation
from sciona.cornell_sampling import sample_waveform
from sciona.cornell_training import train_phase


def main():
    torch.set_num_threads(2)
    source=SourceModel('/private/tmp/sciona_cornell_source')
    waveforms=[(.2*np.sin(np.arange(960001)*f)).astype(np.float32) for f in (.021,.031)]
    noise=(.15*np.sin(np.arange(32000)*.13)+.04).astype(np.float32)
    labels=np.zeros((2,1,264),dtype=np.float32);labels[0,0,0]=1;labels[1,0,1]=1
    reports=[]
    with tempfile.TemporaryDirectory(prefix='cornell-private-training-') as temporary:
        model=source.build(seed=1,synthetic=True)
        for phase,mixup,augmented in [('initial',False,False),('continuation',True,False),('augmented',False,True)]:
            if phase=='augmented':
                del model;gc.collect();model=source.build(seed=3,synthetic=True)
            augmentation=Augmentation('/private/tmp/sciona_cornell_source',
                '/private/tmp/sciona_cornell_dependencies/audiomentations',variant='default',background=[noise],seed=5)
            def batches(epoch):
                rng=np.random.RandomState(100+epoch)
                audio=np.stack([sample_waveform(augmentation(w),training=True,rng=rng) for w in waveforms])
                yield audio,labels,np.zeros_like(labels)
            def validation():
                audio=sample_waveform(waveforms[0],training=False,rng=np.random.RandomState(1))[None]
                truth=np.repeat(labels[:1],2,axis=1)
                yield audio,truth,np.zeros_like(truth)
            result=train_phase(source,model,training_batches=batches,validation_batches=validation,
                steps_per_epoch=1,epochs=2,peak=.0005 if mixup else .001,mixup=mixup,
                augmented_loss=augmented,seed=7,checkpoint_directory=Path(temporary)/phase)
            assert result['selected_epoch']==2 and result['retained_epochs']==[2]
            checkpoint=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,checkpoint['model'][key],rtol=0,atol=0)
            assert checkpoint['iteration']==2 and checkpoint['optimizer']['state']
            reports.append(dict(phase=phase,full_30_second_training=True,validation_two_clips=True,
                optimizer_steps=2,selected_epoch=2,complete_checkpoint_reload=True))
            del checkpoint;gc.collect()
            print(json.dumps(reports[-1]),flush=True)
    names=['cornell_model.py','cornell_training.py','cornell_schedule.py','cornell_sampling.py','cornell_augmentation.py','cornell_validation.py']
    report=dict(status='passed',phases=reports,runtime_sha256={n:hashlib.sha256((ROOT/'sciona'/n).read_bytes()).hexdigest() for n in names},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Three full-length synthetic phases, including model-only continuation and augmented loss; one batch per epoch, random offline initial weights. Not thirteen-member population/competition accuracy evidence.')
    (ROOT/'docs/reviews/competition_cornell_training_execution.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
