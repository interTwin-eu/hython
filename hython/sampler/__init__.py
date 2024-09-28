import numpy as np
import xarray as xr

from torch.utils.data import Dataset
from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import SubsetRandomSampler, DistributedSampler, SequentialSampler, RandomSampler


class SubsetSequentialSampler:
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, indexes) -> None:
        self.indices = indexes

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class SamplerBuilder(TorchSampler):
    def __init__(
        self,
        pytorch_dataset:Dataset, 
        sampling: str = "random",
        sampling_kwargs: dict = {},
        processing: str = "single-gpu",
    ):

        self.dataset = pytorch_dataset

        self.sampling = sampling 

        self.sampling_kwargs = sampling_kwargs

        self.processing = processing


    def get_sampler(self):
        if self.processing == "single-gpu":
            if self.sampling == "random":
                return RandomSampler(self.dataset, **self.sampling_kwargs)
            elif self.sampling == "sequential":
                return SequentialSampler(self.dataset, **self.sampling_kwargs)
        if self.processing == "multi-gpu":
            if self.sampling == "random":
                return DistributedSampler(self.dataset, shuffle=True, **self.sampling_kwargs)
            elif self.sampling == "sequential":
                return DistributedSampler(self.dataset, shuffle=False, **self.sampling_kwargs)
            

from .downsampler import *