"""Compare full dropped-tail batches and main RNG advancement with old source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import BatchSampler, Sampler, RandomSampler

from sciona.wheat_sampler import WheatRandomSampler


def main(source, output):
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!='3e58a2bc4bb221449f18b0db7e965513a107d13485e2b78839003bb5c6199cfd':
        raise ValueError('historical sampler source drift')
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name in {'RandomSampler','BatchSampler'}]
    namespace=dict(torch=torch,Sampler=Sampler,_int_classes=(int,))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<historical-samplers>','exec'),namespace)
    cases=0;modern_difference=False
    for count in (21,40,48,67):
        population=list(range(count))
        for batch_size in (4,8,12,20):
            for seed in range(16):
                torch.manual_seed(seed)
                actual=list(BatchSampler(WheatRandomSampler(population),batch_size,True))
                state=torch.get_rng_state()
                torch.manual_seed(seed)
                expected=list(namespace['BatchSampler'](namespace['RandomSampler'](population),batch_size,True))
                assert actual==expected
                torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
                assert len(actual)==count//batch_size and all(len(batch)==batch_size for batch in actual)
                torch.manual_seed(seed)
                installed=list(BatchSampler(RandomSampler(population),batch_size,True))
                modern_difference |= installed!=expected or not torch.equal(state,torch.get_rng_state())
                cases+=1
    assert modern_difference
    files=['sciona/wheat_sampler.py','scripts/validate_wheat_sampler.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=digest,
        exact_batch_and_rng_cases=cases,all_source_batch_sizes_exercised=True,installed_default_difference_observed=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Sampler and dropped-tail batching only; loader iterator seeding, worker random states and full fits remain separate.',
                'Historical Python algorithm uses installed Torch randperm; old native RNG algorithm parity is not claimed.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_batch_and_rng_cases=cases)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.output)
