"""Source PyTorch 1.4 random sampling without the modern extra seed draw."""
import torch
from torch.utils.data import Sampler


class WheatRandomSampler(Sampler):
    def __init__(self, data_source):
        if len(data_source) < 1:
            raise ValueError('nonempty source population required')
        self.data_source=data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)
