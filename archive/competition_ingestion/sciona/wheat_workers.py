"""Spawn-compatible worker setup preserving historical inherited random states."""
import copy
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from sciona.wheat_augmentation_compat import historical_augmentation_api
from sciona.wheat_dataset import WheatDataset


@dataclass
class WheatWorkerInit:
    numpy_state: tuple
    imgaug_state: tuple

    @classmethod
    def capture(cls):
        """Capture immediately before each loader iterator, as at source fork."""
        with historical_augmentation_api():
            import imgaug
            return cls(copy.deepcopy(np.random.get_state()),copy.deepcopy(imgaug.current_random_state().get_state()))

    def __call__(self,worker_id):
        worker=get_worker_info()
        if worker is None or worker.id!=worker_id:
            raise ValueError('loader worker context required')
        with historical_augmentation_api():
            import imgaug
            import cv2
            cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
            np.random.set_state(self.numpy_state)
            imgaug.current_random_state().set_state(self.imgaug_state)
        torch.set_num_threads(1)
        random.seed(worker.seed)
        torch.manual_seed(worker.seed)


class WheatWorkerDataset(Dataset):
    """Serialize runtime metadata/loader without importing old transforms during unpickling."""
    def __init__(self,dataset):
        self.image_ids=list(dataset.image_ids)
        self.loader=dataset.loader
        self.image_size=dataset.image_size
        self.mode=dataset.mode
        self.network=dataset.network
        self._dataset=None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,index):
        with historical_augmentation_api():
            import imgaug
            if self._dataset is None:
                # Construction already happened before the source's worker fork.
                # Recreate only the stateless transform objects without new RNG draws.
                states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),imgaug.current_random_state().get_state())
                try:
                    self._dataset=WheatDataset(self.image_ids,self.loader,self.image_size,'valid',self.network)
                    self._dataset.image_ids=list(self.image_ids)
                    self._dataset.mode=self.mode
                finally:
                    random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
                    imgaug.current_random_state().set_state(states[3])
            return self._dataset[index]

    def __getstate__(self):
        state=self.__dict__.copy()
        state['_dataset']=None
        return state


def collate_wheat(batch):
    return tuple(zip(*batch))


def source_loader_iterator(loader):
    """Recreate source workers with current inherited states for each epoch."""
    if loader.num_workers!=16 or loader.persistent_workers:
        raise ValueError('source loader requires 16 fresh workers per iterator')
    loader.worker_init_fn=WheatWorkerInit.capture()
    return iter(loader)
