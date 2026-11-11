"""Full B0 synthetic execution at the winner writeup's classifier epoch budget."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from sciona.bengali_classifier_fit import fit_font_classifier
from sciona.bengali_font_classifier import BengaliFontClassifier
from sciona.bengali_population import ImagePopulation
from sciona.bengali_preprocessing import prepare_handwriting
from sciona.bengali_pretraining_sampling import PretrainingParameterSampler


def main(runtime,output):
    if runtime.exists():raise ValueError('fresh private runtime required')
    modules=['classifier_fit','font_classifier','population','preprocessing','pretraining_sampling',
             'pretraining_augmentation','font_sampling','font_augmentation','shear_sampling',
             'sampling','joint_labels','label_corrections']
    files=[Path('sciona/bengali_'+name+'.py') for name in modules]+[Path(__file__)]
    bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    torch.set_num_threads(2);torch.manual_seed(1298)
    rng=np.random.default_rng(817)
    labels=np.arange(64,dtype=np.int64)%8
    images=np.empty((64,137,236),dtype=np.uint8)
    for index,label in enumerate(labels):
        images[index]=np.clip(24+int(label)*27+rng.integers(0,12,(137,236)),0,255).astype(np.uint8)
    population=ImagePopulation(images,labels)
    model=BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0'))
    print(json.dumps(dict(started=True,synthetic_only=True,epochs=60,batch_size=32,optimizer_updates=120)),flush=True)
    result=fit_font_classifier(model,population,epochs=60,batch_size=32,train_seed=831,validation_seed=832,
        sampler=PretrainingParameterSampler(python_seed=833,image_seed=834,backend='sfc64'),output_directory=runtime)
    if result['best_checkpoint'] is None:raise ValueError('source best checkpoint absent; final checkpoint cannot substitute')
    assert bindings=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert result['completed_epochs']==60 and result['optimizer_updates']==120
    receipt=result['best_checkpoint'];path=runtime/receipt['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    state=torch.load(path,weights_only=True)
    model.load_state_dict(state);model.eval()
    fresh=BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0')).eval();fresh.load_state_dict(state)
    with torch.no_grad():
        probe=prepare_handwriting(images[:2]);a,b=model(probe),fresh(probe)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert torch.isfinite(a).all()
    report=dict(passed=True,synthetic_only=True,approved=False,catalog_mutations=0,
        epochs=60,batch_size=32,optimizer_updates=120,full_efficientnet_b0=True,
        source_best_checkpoint_present=True,fresh_checkpoint_output_exact=True,implementation_sha256=bindings,
        same_population_accuracy=result['history'][-1]['same_population_accuracy'],
        limits=['Synthetic intensity patterns are not font images; no glyph-recognition or winning-accuracy claim.',
                'Epoch/batch controls match winner writeup; population size is synthetic and does not reproduce original update count.',
                'Same-population evaluation is not held-out validation.',
                'Checkpoint qualifies this synthetic execution only; complete pipeline, provenance and publication remain pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();main(args.runtime_directory,args.output)
