"""Compare update ordering against pinned source using small synthetic networks."""
import argparse
import ast
import copy
import hashlib
import itertools
import json
import random
from pathlib import Path

import torch
from torch import nn

from sciona.bengali_gan_training import CycleGANTraining


class SyntheticGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.7))

    def forward(self, images):
        return images * self.scale


class SyntheticDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3))

    def forward(self, images):
        return images.mean((1,2,3)) * self.scale


def main(source, output):
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if source_hash != '600d64d01226bd5ca879a347ca06f148193fd73c88039222cee1832c847bc716':
        raise ValueError('reference source hash mismatch')
    text = raw.decode()
    namespace = dict(torch=torch, nn=nn, random=random.Random(83))
    for begin,end,names in [('class ImagePool():','class BengalModel(',{'ImagePool','GANLoss'}),
                            ('class CycleGan(nn.Module):','model = CycleGan(',{'CycleGan'})]:
        parsed = ast.parse(text[text.index(begin):text.index(end)])
        nodes = [n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError('missing reference classes')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-training-reference>','exec'),namespace)
    torch.set_num_threads(2)
    torch.manual_seed(193)
    networks = [SyntheticGenerator(),SyntheticGenerator(),SyntheticDiscriminator(),SyntheticDiscriminator(),
                nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,14784))]
    actual = CycleGANTraining(generator_a=networks[0],generator_b=networks[1],
        discriminator_a=networks[2],discriminator_b=networks[3],classifier=networks[4],
        classifier_weight=1.,total_steps=4,replay_seed_a=83,replay_seed_b=84)
    ref_networks = copy.deepcopy(networks)
    reference = namespace['CycleGan'](*ref_networks,namespace['GANLoss']('lsgan'),nn.CrossEntropyLoss(),10.,10.,1.,'cpu')
    g_opt = torch.optim.Adam(itertools.chain(ref_networks[0].parameters(),ref_networks[1].parameters()),lr=.0002,betas=(.5,.999))
    d_opt = torch.optim.Adam(itertools.chain(ref_networks[2].parameters(),ref_networks[3].parameters()),lr=.0002,betas=(.5,.999))
    rate = lambda step: 1. if step < 2 else (4-step)/2
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt,rate) for opt in (g_opt,d_opt)]
    generator_keys = ['total','gan_a','gan_b','cycle_a','cycle_b','font']
    discriminator_keys = ['total','real_a','fake_a','domain_a','real_b','fake_b','domain_b']
    for step in range(4):
        a,b = torch.rand(2,3,224,224),torch.rand(2,3,224,224)
        labels = torch.tensor([step,14783-step])
        result = actual.step(a,b,labels)
        reference.set_input(a,b,labels,labels)
        reference.forward()
        g_opt.zero_grad()
        g_losses = reference.generator_step()
        g_opt.step()
        d_opt.zero_grad()
        d_losses = reference.discriminator_step()
        d_opt.step()
        for scheduler in schedulers:
            scheduler.step()
        for keys,losses,group in [(generator_keys,g_losses,'generator'),(discriminator_keys,d_losses,'discriminator')]:
            for key,loss in zip(keys,losses):
                if result[group][key] != float(loss.detach()):
                    raise AssertionError(f'{group} {key} differs at step {step}')
        for left,right in zip(networks,ref_networks):
            for key,value in left.state_dict().items():
                torch.testing.assert_close(value,right.state_dict()[key],rtol=0,atol=0)
        for pool,refpool in [(actual.pool_a,reference.image_pool_a),(actual.pool_b,reference.image_pool_b)]:
            torch.testing.assert_close(torch.stack(pool.images),torch.cat(refpool.images),rtol=0,atol=0)
    report = dict(passed=True,synthetic_only=True,catalog_mutations=0,compared_updates=4,
        exact_losses_and_network_states=True,exact_replay_history=True,source_only_sha256=source_hash,
        sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['scripts/validate_bengali_training_reference.py','sciona/bengali_gan_training.py','sciona/bengali_gan_losses.py']},
        limits=['Small synthetic networks qualify update mechanics only; full source architectures and winning budgets remain pending.',
                'Replay pools remain below capacity in this probe; separate replay validation covers replacement behavior.',
                'Classifier has no stochastic layers here; separate guidance tests qualify preservation of the source eval-mode behavior.',
                'Both executions use the installed PyTorch runtime; historical runtime equivalence is not established.'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    main(args.source,args.output)
