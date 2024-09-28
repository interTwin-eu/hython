import numpy as np
import xarray as xr
import itertools

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Any, Tuple, List
from numpy.typing import NDArray
from xarray.core.coordinates import DatasetCoordinates

from torch.utils.data import Dataset
from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import SubsetRandomSampler, DistributedSampler, SequentialSampler, RandomSampler


from hython.utils import compute_grid_indices, get_unique_spatial_idxs, get_unique_time_idxs

@dataclass
class SamplerResult:
    """Metadata"""
    # TODO: rename and check if all of them are still necessary
    idx_grid_2d: NDArray
    idx_sampled_1d: NDArray
    idx_sampled_1d_nomissing: NDArray | None
    idx_missing_1d: NDArray | None
    sampled_grid: NDArray | None
    sampled_grid_dims: tuple | None
    xr_sampled_coords: DatasetCoordinates | None

    def __repr__(self):
        return f"SamplerResult(\n - id_grid_2d: {self.idx_grid_2d.shape} \n - idx_sampled_1d: {self.idx_sampled_1d.shape} \n - idx_sampled_1d_nomissing: {self.idx_sampled_1d_nomissing.shape}) \n - idx_missing_1d: {self.idx_missing_1d.shape} \n - sampled_grid_dims: {self.sampled_grid_dims} \n - xr_coords: {self.xr_sampled_coords}"

# === DOWNSAMPLERS ===============================================================
class AbstractDownSampler(ABC):
    def __init__(self):
        """Pass parametes required by the downsampling approach"""
        pass

    def compute_grid_indices(self, shape=None, grid=None):
        if shape is not None:
            return compute_grid_indices(shape=shape)
        elif grid is not None:
            return compute_grid_indices(grid=grid)
        else:
            raise Exception("Provide either shape or grid")

    @abstractmethod
    def sampling_idx(
        self, shape: tuple[int], grid: NDArray | xr.DataArray | xr.Dataset
    ) -> SamplerResult:
        """Sample the original grid. Must be instantiated by a concrete class that implements the sampling approach.

        Args:
            grid (NDArray | xr.DataArray | xr.Dataset): The gridded data to be sampled

        Returns:
            Tuple[NDArray, SamplerMetaData]: The sampled grid and sampler's metadata
        """

        pass


class CubeletsDownsampler(AbstractDownSampler):
    def __init__(self, temporal_downsample_fraction: float = 0.5, spatial_downsample_fraction: float = 0.5):
        self.temporal_frac = temporal_downsample_fraction
        self.spatial_frac = spatial_downsample_fraction

    def sampling_idx(self, indexes):
        
        idxs_sampled = {}
        
        time_idx = get_unique_time_idxs(indexes)
        spatial_idx = get_unique_spatial_idxs(indexes)

        time_sub_idx = np.random.choice(time_idx, size=int(self.temporal_frac*len(time_idx)), replace=False)

        spatial_sub_idx = np.random.choice(spatial_idx, size=int(self.spatial_frac*len(spatial_idx)), replace=False)
        
        for filter in itertools.product(spatial_sub_idx, time_sub_idx):

            value = indexes.get(filter, None)
            if value is not None:
                idxs_sampled[filter] = value
                
        return idxs_sampled
class RegularIntervalDownsampler(AbstractDownSampler):
    def __init__(self, intervals: list[int], origin: list[int]):
        self.intervals = intervals
        self.origin = origin

        if intervals[0] != intervals[1]:
            raise NotImplementedError("Different x,y intervals not yet implemented!")

        if origin[0] != origin[1]:
            raise NotImplementedError("Different x,y origins not yet implemented!")

    def sampling_idx(
        self, indexes, shape
    ):  # remove missing is a 2D mask
        """Sample a N-dimensional array by regularly-spaced points along the spatial axes.

        mask_missing, removes missing values from grid where mask is True
        """

        xr_coords = None
        sampled_grid = None
        sampled_grid_dims = None

        idx_nan = np.array([])

        ishape, iorigin, iintervals = (
            shape[0],
            self.origin[0],
            self.intervals[0],
        )  # rows (y, lat)
        
        jshape, jorigin, jintervals = (
            shape[1],
            self.origin[1],
            self.intervals[1],
        )  # columns (x, lon)

        irange = np.arange(iorigin, ishape, iintervals)
        jrange = np.arange(jorigin, jshape, jintervals)

        idxs_sampled = indexes[irange[:, None], jrange].flatten()  # broadcasting


        return idxs_sampled


# === TRAINING SAMPLERS ===============================================================
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





# === SAMPLER BUILDER =========================================================

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

