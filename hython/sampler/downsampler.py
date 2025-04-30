from . import *

from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools

from typing import Any, Tuple, List
from numpy.typing import NDArray

from hython.utils import (
    compute_grid_indices,
    get_unique_spatial_idxs,
    get_unique_time_idxs,
    downsample_spacetime,
    downsample_time,
    downsample_space
)


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
    ):
        """Sample the original grid. Must be instantiated by a concrete class that implements the sampling approach.

        Args:
            grid (NDArray | xr.DataArray | xr.Dataset): The gridded data to be sampled

        Returns:
            Tuple[NDArray, SamplerMetaData]: The sampled grid and sampler's metadata
        """

        pass


class RandomDownsampler(AbstractDownSampler):
    def __init__(
        self,
        frac_time: float | None = 0.5,
        frac_space: float | None = 0.5,  
    ):
        self.frac_time = frac_time 
        self.frac_space = frac_space

    def sampling_idx(self, coords):
        space, time = coords
        if self.frac_time:
            time = np.sort(np.random.choice(time, int(len(time)*self.frac_time), replace=False))
        if self.frac_space:
            space = np.sort(np.random.choice(space, int(len(space)*self.frac_space), replace=False))
        return [space, time]


class CubeletsDownsampler(AbstractDownSampler):
    def __init__(
        self,
        temporal_downsample_fraction: float = 0.5,
        spatial_downsample_fraction: float = 0.5,
    ):
        self.temporal_frac = temporal_downsample_fraction
        self.spatial_frac = spatial_downsample_fraction

    def sampling_idx(self, indexes):
        idxs_sampled = {}

        time_idx = get_unique_time_idxs(indexes)
        spatial_idx = get_unique_spatial_idxs(indexes)

        time_sub_idx = np.random.choice(
            time_idx, size=int(self.temporal_frac * len(time_idx)), replace=False
        )

        spatial_sub_idx = np.random.choice(
            spatial_idx, size=int(self.spatial_frac * len(spatial_idx)), replace=False
        )

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

    def sampling_idx(self, indexes, shape):  # remove missing is a 2D mask
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


