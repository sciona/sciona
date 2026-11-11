"""Full-sized synthetic CycleGAN integration at the notebook demo batch size."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet

from sciona.bengali_fit import fit_population
from sciona.bengali_font_classifier import BengaliFontClassifier
from sciona.bengali_font_sampling import FontParameterSampler
from sciona.bengali_gan_networks import BengaliGenerator,BengaliDiscriminator,initialize_image_network
from sciona.bengali_gan_training import CycleGANTraining
from sciona.bengali_population import ImagePopulation
from sciona.bengali_preprocessing import prepare_handwriting
from sciona.bengali_sampling import SamplingBudget


def main(runtime,output):
    if runtime.exists():
        raise ValueError('fresh private runtime directory required')
    torch.set_num_threads(2)
    torch.manual_seed(831)
    rng=np.random.default_rng(532)
    images=rng.integers(0,256,(16,137,236),dtype=np.uint8)
    hand=ImagePopulation(images,np.arange(16))
    font=ImagePopulation(rng.integers(0,256,(16,137,236),dtype=np.uint8),np.arange(16))
    networks=[BengaliGenerator(),BengaliGenerator(),BengaliDiscriminator(),BengaliDiscriminator()]
    for index,network in enumerate(networks):
        initialize_image_network(network,generator=torch.Generator().manual_seed(731+index))
    classifier=BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0'))
    frozen={k:v.clone() for k,v in classifier.state_dict().items()}
    before=[next(n.parameters()).detach().clone() for n in networks]
    budget=SamplingBudget(16,16,8,1)
    trainer=CycleGANTraining(generator_a=networks[0],generator_b=networks[1],discriminator_a=networks[2],
        discriminator_b=networks[3],classifier=classifier,classifier_weight=1.,total_steps=budget.total_steps,
        replay_seed_a=14,replay_seed_b=15)
    print(json.dumps(dict(started=True,synthetic_only=True,epochs=1,batch_size=8,steps=2)),flush=True)
    result=fit_population(trainer,hand,font,budget,hand_seed=39,font_seed=40,
        font_sampler=FontParameterSampler(python_seed=41,image_seed=42,backend='sfc64'),output_directory=runtime)
    assert all(not torch.equal(old,next(n.parameters())) for old,n in zip(before,networks))
    assert all(torch.isfinite(p).all() for n in networks for p in n.parameters())
    for name,value in classifier.state_dict().items():
        torch.testing.assert_close(value,frozen[name],rtol=0,atol=0)
    receipt=result['history'][-1]
    path=runtime/receipt['generator_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['generator_sha256']
    reloaded=BengaliGenerator().eval()
    reloaded.load_state_dict(torch.load(path,weights_only=True))
    with torch.no_grad():
        probe=prepare_handwriting(images[:1])
        torch.testing.assert_close(reloaded(probe),trainer.generator_b(probe),rtol=0,atol=0)
    report=dict(passed=True,synthetic_only=True,catalog_mutations=0,approved=False,
        completed_epochs=result['completed_epochs'],completed_updates=result['completed_steps'],batch_size=8,
        all_four_trainable_networks_updated=True,frozen_classifier_unchanged=True,
        fresh_checkpoint_output_exact=True,epoch_metrics=receipt['metrics'],
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('sciona').glob('bengali_*.py'))},
        limits=['Notebook demonstration epoch/batch settings only; paper winning training budget is40epochs batch32.',
                'Synthetic images and random untrained B0 classifier; no font-recognition or winning accuracy claim.',
                'Explicit reconstruction seeds and SFC64 backend; notebook worker seeding and historical environment remain unqualified.',
                'Complete seen/OOD/font-pretraining/inference pipeline and promotion gates remain pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['sha256','epoch_metrics']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.runtime_directory,args.output)
