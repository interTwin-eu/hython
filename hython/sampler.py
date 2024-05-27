import torch
import numpy as np
import xarray as xr

from abc import ABC, abstractmethod
from dataclasses import dataclass


# type hints
from typing import Any, Tuple, List
from numpy.typing import NDArray
from xarray.core.coordinates import DatasetCoordinates

from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import RandomSampler
from torch.utils.data import SubsetRandomSampler, SequentialSampler

from hython.utils import missing_location_idx


def get_grid_idx(shape=None, grid=None):
    if grid is not None:
        if isinstance(grid, np.ndarray):
            shape = grid.shape
        elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
            shape = (len(grid.lat), len(grid.lon))
        else:
            pass

    ishape = shape[0]  # rows (y, lat)
    jshape = shape[1]  # columns (x, lon)

    grid_idx = np.arange(0, ishape * jshape, 1).reshape(ishape, jshape)

    return grid_idx


def torch_dataset_to_xygrid(dataset):
    return


@dataclass
class SamplerResult:
    """Store metadata to restructure original grid from the sampled grid"""

    idx_grid_2d: NDArray
    idx_sampled_1d: NDArray
    idx_sampled_1d_nomissing: NDArray | None
    idx_missing_1d: NDArray | None
    sampled_grid: NDArray | None
    sampled_grid_dims: tuple | None
    xr_sampled_coords: DatasetCoordinates | None

    def __repr__(self):
        return f"SamplerResult(\n - id_grid_2d: {self.idx_grid_2d.shape} \n - idx_sampled_1d: {self.idx_sampled_1d.shape} \n - idx_sampled_1d_nomissing: {self.idx_sampled_1d_nomissing.shape}) \n - idx_missing_1d: {self.idx_missing_1d.shape} \n - sampled_grid_dims: {self.sampled_grid_dims} \n - xr_coords: {self.xr_sampled_coords}"


class AbstractDownSampler(ABC):
    def __init__(self):
        """Pass parametes required by the sampling approach"""
        pass

    def get_grid_idx(self, shape=None, grid=None):
        if shape is not None:
            return get_grid_idx(shape=shape)
        elif grid is not None:
            return get_grid_idx(grid=grid)
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


class RegularIntervalDownSampler(AbstractDownSampler):
    def __init__(self, intervals: list[int] = (5, 5), origin: list[int] = (0, 0)):
        self.intervals = intervals
        self.origin = origin

        if intervals[0] != intervals[1]:
            raise NotImplementedError("Different x,y intervals not yet implemented!")

        if origin[0] != origin[1]:
            raise NotImplementedError("Different x,y origins not yet implemented!")

    def sampling_idx(
        self, shape, missing_mask=None, grid=None
    ):  # remove missing is a 2D mask
        """Sample a N-dimensional array by regularly-spaced points along the spatial axes.

        Parameters
        ----------
        grid : np.ndarray
            Spatial axes should be the first 2 dimensions, i.e. (lat, lon) or (y, x)
        intervals : tuple[int], optional
            Sampling intervals in CRS distance, by default (5,5).
            5,5 in a 1 km resolution grid, means sampling every 5 km in x and y directions.
        origin : tuple[int], optional
            _description_, by default (0, 0)

        Returns
        -------
        np.ndarray
            _description_
        """

        xr_coords = None
        sampled_grid = None
        sampled_grid_dims = None

        idx_nan = np.array([])

        if isinstance(grid, np.ndarray):
            shape = grid.shape
        elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
            shape = (len(grid.lat), len(grid.lon))
        else:
            pass

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

        if shape is not None:
            grid_idx = self.get_grid_idx(shape=shape)
        else:
            grid_idx = self.get_grid_idx(grid=grid)

        idx_sampled = grid_idx[irange[:, None], jrange].flatten()  # broadcasting

        if missing_mask is not None:
            idx_nan = grid_idx[missing_mask]

            print(idx_nan)

            idx_sampled_1d_nomissing = np.setdiff1d(idx_sampled, idx_nan)
        else:
            idx_sampled_1d_nomissing = idx_sampled

        if grid is not None:
            if isinstance(grid, np.ndarray):
                sampled_grid = grid[irange[:, None], jrange]
            elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
                sampled_grid = grid.isel(lat=irange, lon=jrange)

                xr_coords = sampled_grid.coords
            else:
                pass

            sampled_grid_dims = sampled_grid.shape  # lat, lon

        return SamplerResult(
            idx_grid_2d=grid_idx,
            idx_sampled_1d=idx_sampled,
            idx_sampled_1d_nomissing=idx_sampled_1d_nomissing,
            idx_missing_1d=idx_nan,
            sampled_grid=sampled_grid,
            sampled_grid_dims=sampled_grid_dims,
            xr_sampled_coords=xr_coords,
        )


class NoDownsampling(AbstractDownSampler):
    def __init__(self, replacement: bool):
        pass

    def sampling_idx(
        self, shape, missing_mask=None, grid=None
    ):  # remove missing is a 2D mask
        """Sample a N-dimensional array by regularly-spaced points along the spatial axes.

        Parameters
        ----------
        grid : np.ndarray
            Spatial axes should be the first 2 dimensions, i.e. (lat, lon) or (y, x)
        intervals : tuple[int], optional
            Sampling intervals in CRS distance, by default (5,5).
            5,5 in a 1 km resolution grid, means sampling every 5 km in x and y directions.
        origin : tuple[int], optional
            _description_, by default (0, 0)

        Returns
        -------
        np.ndarray
            _description_

        Returns all indices but the missing values
        Prepare for SubsetRandomSampler
        In case no missing then RandomSampler
        """

        xr_coords = None
        sampled_grid = None
        sampled_grid_dims = None

        idx_nan = np.array([])

        if isinstance(grid, np.ndarray):
            shape = grid.shape
        elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
            shape = (len(grid.lat), len(grid.lon))
        else:
            pass

        ishape = shape[0]  # rows (y, lat)
        jshape = shape[1]  # columns (x, lon)

        irange = np.arange(0, ishape, 1)
        jrange = np.arange(0, jshape, 1)

        if shape is not None:
            grid_idx = self.get_grid_idx(shape=shape)
        else:
            grid_idx = self.get_grid_idx(grid=grid)

        idx_sampled = grid_idx[irange[:, None], jrange].flatten()  # broadcasting

        if missing_mask is not None:
            idx_nan = grid_idx[missing_mask]

            idx_sampled_1d_nomissing = np.setdiff1d(idx_sampled, idx_nan)
        else:
            idx_sampled_1d_nomissing = idx_sampled

        if grid is not None:
            if isinstance(grid, np.ndarray):
                sampled_grid = grid[irange[:, None], jrange]
            elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
                sampled_grid = grid.isel(lat=irange, lon=jrange)

                xr_coords = sampled_grid.coords
            else:
                pass

            sampled_grid_dims = sampled_grid.shape  # lat, lon

        return SamplerResult(
            idx_grid_2d=grid_idx,
            idx_sampled_1d=idx_sampled,
            idx_sampled_1d_nomissing=idx_sampled_1d_nomissing,
            idx_missing_1d=idx_nan,
            sampled_grid=sampled_grid,
            sampled_grid_dims=sampled_grid_dims,
            xr_sampled_coords=xr_coords,
        )


class SubsetSequentialSampler:
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, indices) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


CLASS_METHODS = {
    "regular": RegularIntervalDownSampler,  # method to get indices through method "sampling_idx", then use SubsetRandomSample or SubsetSequentialSample
    "default": NoDownsampling,
}


class SamplerBuilder(TorchSampler):
    def __init__(
        self,
        downsampling: bool = False,
        downsampling_method: str = "regular",
        downsampling_method_kwargs: dict = {},
        sampling: str = "random",
        sampling_kwargs: dict = {},
        processing: str = "single-gpu",
    ):
        if downsampling:
            # downsampling
            self.method = downsampling_method
            self.method_kwargs = downsampling_method_kwargs
            self.method_class = CLASS_METHODS.get(self.method, False)
        else:
            # no downsampling
            self.method = "default"
            self.method_class = CLASS_METHODS.get(self.method, False)

        self.sampling = sampling
        self.sampling_kwargs = sampling_kwargs
        self.processing = processing

    def initialize(self, shape=None, mask_missing=None, grid=None, torch_dataset=None):
        """ """

        self.mask_missing = mask_missing

        self.grid = grid  # 2d grid
        self.torch_dataset = torch_dataset

        self.method_instance = self.method_class(**self.method_kwargs)

        self.result = self.method_instance.sampling_idx(
            shape, self.mask_missing, self.grid
        )

        found_missing = len(self.result.idx_missing_1d) > 0

        if found_missing:
            print("found missing")
            self.indices = self.result.idx_sampled_1d_nomissing.tolist()
        else:
            print("not found missing")
            self.indices = self.result.idx_sampled_1d.tolist()

    def get_sampler(self):
        if self.processing == "single-gpu":
            if self.sampling == "random":
                return SubsetRandomSampler(self.indices)
            elif self.sampling == "sequential":
                return SubsetSequentialSampler(self.indices)
        if self.processing == "multi-gpu":
            raise NotImplementedError()

    def get_metadata(self):
        return self.result
