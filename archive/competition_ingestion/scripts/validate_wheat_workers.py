"""Exercise 16 spawned workers against isolated historical-state replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,get_worker_info

from sciona.wheat_augmentation_compat import historical_augmentation_api
from sciona.wheat_dataset import WheatDataset
from sciona.wheat_sampler import WheatRandomSampler
from sciona.wheat_workers import WheatWorkerDataset,WheatWorkerInit,source_loader_iterator


class SyntheticLoader:
    def __call__(self,key):
        width=1408 if key%2 else 1024
        image=np.full((1024,width,3),32+(int(key)%8)*23,dtype=np.uint8)
        image[200:800,200:900,:]=220
        return image,np.array([[200.,200.,900.,800.]]),'spike' if key%2 else 'primary'


def receipt(sample,index,worker_id,seed):
    image,target=sample
    digest=hashlib.sha256(image.contiguous().numpy().tobytes())
    for key in sorted(target):
        digest.update(key.encode());digest.update(target[key].contiguous().numpy().tobytes())
    with historical_augmentation_api():
        import imgaug
        draws=[random.random(),float(np.random.random()),float(torch.rand(())),float(imgaug.current_random_state().random_sample())]
    return dict(index=index,worker=worker_id,seed=seed,digest=digest.hexdigest(),draws=draws)


class ProbeDataset(Dataset):
    def __init__(self,dataset):self.dataset=WheatWorkerDataset(dataset)
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        info=get_worker_info()
        return receipt(self.dataset[index],index,info.id,info.seed)


def collate(batch):return batch


def main(output):
    torch.set_num_threads(1)
    with historical_augmentation_api():
        import imgaug
        import cv2
        cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
        random.seed(1537);np.random.seed(1537);imgaug.seed(1537)
        original=WheatDataset(list(range(128)),SyntheticLoader(),512)
        torch.manual_seed(1538)
        loader=DataLoader(ProbeDataset(original),batch_size=4,sampler=WheatRandomSampler(original),
            drop_last=True,num_workers=16,multiprocessing_context='spawn',
            collate_fn=collate,persistent_workers=False,prefetch_factor=2)
        for epoch in range(2):
            references=[copy.deepcopy(original) for _ in range(16)]
            prior_torch=torch.get_rng_state()
            actual=list(source_loader_iterator(loader))
            initializer=loader.worker_init_fn
            parent_states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),imgaug.current_random_state().get_state())
            print(json.dumps(dict(epoch=epoch,spawned_workers_completed=16,batches=len(actual))),flush=True)
            torch.set_rng_state(prior_torch)
            base_seed=torch.empty((),dtype=torch.int64).random_().item()
            order=torch.randperm(len(original)).tolist()
            torch.testing.assert_close(parent_states[2],torch.get_rng_state(),rtol=0,atol=0)
            states=[]
            for worker_id in range(16):
                seed=base_seed+worker_id
                states.append((random.Random(seed).getstate(),copy.deepcopy(initializer.numpy_state),
                               torch.Generator().manual_seed(seed).get_state(),copy.deepcopy(initializer.imgaug_state)))
            for batch_index,start in enumerate(range(0,len(order),4)):
                worker_id=batch_index%16;seed=base_seed+worker_id
                py,numpy_state,torch_state,ia_state=states[worker_id]
                random.setstate(py);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)
                imgaug.current_random_state().set_state(ia_state)
                expected=[receipt(references[worker_id][index],index,worker_id,seed) for index in order[start:start+4]]
                assert actual[batch_index]==expected, f'worker replay differs for epoch {epoch} batch {batch_index}'
                states[worker_id]=(random.getstate(),np.random.get_state(),torch.get_rng_state(),imgaug.current_random_state().get_state())
            random.setstate(parent_states[0]);np.random.set_state(parent_states[1]);torch.set_rng_state(parent_states[2])
            imgaug.current_random_state().set_state(parent_states[3])
            # Parent-side training consumes RNG between source loader epochs.
            random.random();np.random.beta(1.,1.);torch.rand(7);imgaug.current_random_state().random_sample()
    files=['sciona/wheat_workers.py','sciona/wheat_dataset.py','sciona/wheat_sampler.py','sciona/wheat_augmentation.py',
           'sciona/wheat_augmentation_compat.py','sciona/wheat_crop.py','sciona/wheat_mosaic.py','scripts/validate_wheat_workers.py']
    report=dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,workers=16,epochs=2,batches=64,samples=256,
        full_augmented_tensors_targets_and_four_rng_streams_exact=True,main_sampler_rng_exact=True,
        implementation_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        limits=['Spawned workers compared with isolated historical-seed replay using the separately qualified dataset implementation.',
                'Two loader epochs with parent RNG advancement; full source fits remain pending.',
                'Installed native libraries; historical operating-system process behavior is not claimed.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,workers=16,epochs=2,batches=64,samples=256)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
