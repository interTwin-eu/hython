import numpy as np
import xarray as xr
import random

from hython.utils import generate_time_idx

from torch.utils.data import Dataset
from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import (
    SubsetRandomSampler,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
)



class RandomTemporalDynamicDownsampler(RandomSampler):
    """Every epoch generate a random subset of temporal indices.
    The generated indices (idx) are used by the dataloader to sample the dataset getitem[idx] """
    def __init__(self, data_source, dynamic_downsampler, replacement=False):
        super(RandomTemporalDynamicDownsampler, self).__init__(data_source)

        self.data_source = data_source
        self.replacement = replacement
        self.spacetime_index = self.data_source.spacetime_index
        self.cell_size  = len(self.data_source.cell_coords[self.data_source.cell_linear_index ])
        self.seq_len = self.data_source.seq_len
        self.time_size = self.data_source.time_size

        frac_time = dynamic_downsampler.get("frac_time")
        self.temporal_subset_size = int( (self.time_size -self.seq_len)*frac_time)

        
        # the total samples
        self.total_subset_size = self.temporal_subset_size*self.cell_size

    def __iter__(self):
        if self.replacement:
            self.time_indices = np.random.randint(0, self.time_size - self.seq_len, self.temporal_subset_size)
        else:
            self.time_indices = random.sample(range(self.time_size - self.seq_len), self.temporal_subset_size)
            
        indeces = generate_time_idx(self.time_indices, self.time_size - self.seq_len, self.seq_len, self.cell_size )

        return iter(indeces)
    
    def __len__(self):
        return self.total_subset_size
    
class SequentialTemporalDynamicDownsampler(RandomSampler):
    """Every epoch generate a random subset of temporal indices.
    The generated indices (idx) are used by the dataloader to sample the dataset getitem[idx] """
    def __init__(self, data_source, dynamic_downsampler, replacement=False):
        super(SequentialTemporalDynamicDownsampler, self).__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.spacetime_index = self.data_source.spacetime_index
        self.cell_size  = len(self.data_source.cell_coords[self.data_source.cell_linear_index ])
        self.seq_len = self.data_source.seq_len
        
        self.time_size = self.data_source.time_size

        frac_time = dynamic_downsampler.get("frac_time")
        self.temporal_subset_size = int( (self.time_size -self.seq_len)*frac_time)

        # the total samples
        self.total_subset_size = self.temporal_subset_size*self.cell_size
        print(self.total_subset_size)

    def __iter__(self):
        if self.replacement:
            time_indices = np.random.randint(0, self.time_size - self.seq_len, self.temporal_subset_size)
        else:
            time_indices = random.sample(range(self.time_size - self.seq_len), self.temporal_subset_size)
            
        indeces = generate_time_idx(time_indices, self.time_size - self.seq_len,self.seq_len, self.cell_size )

        return iter(indeces)

    def __len__(self):
        return self.total_subset_size
    
class DistributedTemporalDynamicDownsampler(DistributedSampler):
    """Every epoch generate a random subset of temporal indices.
    The generated indices (idx) are used by the dataloader to sample the dataset getitem[idx] """
    def __init__(self, data_source, dynamic_downsampler, shuffle = True, replacement=False, sampling_kwargs={}):
        super(DistributedTemporalDynamicDownsampler, self).__init__(
            dataset=data_source, 
            shuffle= shuffle,
            **sampling_kwargs)
                   
        self.data_source = data_source
        self.replacement = replacement
        self.spacetime_index = self.data_source.spacetime_index
        self.cell_size  = len(self.data_source.cell_coords[self.data_source.cell_linear_index])
        self.seq_len = self.data_source.seq_len
        
        self.time_size = self.data_source.time_size

        frac_time = dynamic_downsampler.get("frac_time")
        self.temporal_subset_size = int( (self.time_size -self.seq_len)*frac_time)
        # the total samples
        self.total_subset_size = self.temporal_subset_size*self.cell_size

    def __iter__(self):
        if self.replacement:
            time_indices = np.random.randint(0, self.time_size - self.seq_len, self.temporal_subset_size)
        else:
            time_indices = random.sample(range(self.time_size - self.seq_len), self.temporal_subset_size)
            
        ind = generate_time_idx(time_indices, self.time_size - self.seq_len, self.seq_len, self.cell_size )

        return iter(ind)
    
    def __len__(self):
        return self.total_subset_size
    
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
        cfg,
        pytorch_dataset: Dataset,
        sampling: str = "random",
        sampling_kwargs: dict = {},
        processing: str = "single-gpu",
    ):
        
        self.cfg = cfg 

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
            elif self.sampling == "temporal-downsampling-random":
                return RandomTemporalDynamicDownsampler(
                    self.dataset, self.cfg.dynamic_downsampler,  **self.sampling_kwargs
                )
            elif self.sampling == "temporal-downsampling-sequential":
                return SequentialTemporalDynamicDownsampler(
                    self.dataset, self.cfg.dynamic_downsampler, **self.sampling_kwargs
                )

        if self.processing == "multi-gpu":
            if self.sampling == "random":
                return DistributedSampler(
                    self.dataset, shuffle=True, **self.sampling_kwargs
                )
            elif self.sampling == "sequential":
                return DistributedSampler(
                    self.dataset, shuffle=False, **self.sampling_kwargs
                )
            elif self.sampling == "temporal-downsampling-random":
                return DistributedTemporalDynamicDownsampler(
                    self.dataset, self.cfg.dynamic_downsampler, shuffle=True, **self.sampling_kwargs
                )
            elif self.sampling == "temporal-downsampling-sequential":
                return DistributedTemporalDynamicDownsampler(
                    self.dataset, self.cfg.dynamic_downsampler, shuffle=False, **self.sampling_kwargs
                )


from .downsampler import *
