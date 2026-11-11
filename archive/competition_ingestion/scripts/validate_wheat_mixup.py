"""Compare source mixup arrays, target concatenation and three RNG states."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from sciona.wheat_mixup import mixup_batch


def main(root, output):
    cases = 0
    for detector, file, digest in [
        ('effdet','effdet_train.py','d469f63af50abdb8a9e704bb756a569fd5f06383d3a5d09bd36b951220dbec4c'),
        ('fasterrcnn','faster_rcnn_fpn_train.py','46086931f010b57fdb31e0f613dfa4fdd4a76461d39c742edd288144dbc5a588')]:
        raw=(root/file).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('historical mixup source drift')
        node=next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.If)
            and ast.unparse(n.test)=='random.random() > 0.5 and epoch >= args.warm_epochs')
        class CPUDispatch(ast.NodeTransformer):
            def visit_Call(self, node):
                node=self.generic_visit(node)
                if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda' and not node.args and not node.keywords:
                    return node.func.value
                return node
        code=compile(ast.fix_missing_locations(ast.Module(body=[CPUDispatch().visit(node)],type_ignores=[])), '<historical-mixup-cpu>', 'exec')
        for epoch in (0,19,20,99):
            for seed in range(32):
                images=[torch.full((3,7,9), float(i)/4) for i in range(4)]
                targets=[dict(boxes=torch.tensor([[float(i),1.,4.,5.]]),labels=torch.ones(1,dtype=torch.int64),
                    area=torch.tensor([float(4-i)*4]),iscrowd=torch.zeros(1,dtype=torch.int64)) for i in range(4)]
                before=copy.deepcopy(targets)
                actual_py, expected_py=random.Random(seed),random.Random(seed)
                actual_np, expected_np=np.random.RandomState(seed),np.random.RandomState(seed)
                generator=torch.Generator().manual_seed(seed)
                actual_images, actual_targets=mixup_batch(images,targets,epoch,detector,
                    python_rng=actual_py,numpy_rng=actual_np,torch_generator=generator)
                torch.manual_seed(seed)
                namespace=dict(torch=torch,np=SimpleNamespace(clip=np.clip,random=expected_np),random=expected_py,
                    images=images,targets=targets,epoch=epoch,args=SimpleNamespace(warm_epochs=20))
                exec(code,namespace)
                expected_images=namespace['images']
                if not isinstance(expected_images,torch.Tensor):expected_images=torch.stack(expected_images)
                torch.testing.assert_close(actual_images,expected_images,rtol=0,atol=0)
                for a,b in zip(actual_targets,namespace['targets']):
                    assert set(a)==set(b)
                    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                assert actual_py.getstate()==expected_py.getstate()
                a,b=actual_np.get_state(),expected_np.get_state()
                assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
                torch.testing.assert_close(generator.get_state(),torch.get_rng_state(),rtol=0,atol=0)
                for a,b in zip(targets,before):
                    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                cases+=1
    files=['sciona/wheat_mixup.py','scripts/validate_wheat_mixup.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,exact_cases=cases,
        pixels_targets_and_python_numpy_torch_rng_exact=True,warmup_boundary_exercised=True,inputs_unchanged=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Source CUDA placement removed for CPU comparison; source arithmetic and random-call ordering retained.',
                'Batch mixup only; image-level augmentation and full training remain pending.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,exact_cases=cases)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.source_root,args.output)
