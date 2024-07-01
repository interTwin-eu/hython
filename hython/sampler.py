import numpy as np
import xarray as xr

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Any, Tuple, List
from numpy.typing import NDArray
from xarray.core.coordinates import DatasetCoordinates

from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import SubsetRandomSampler


from hython.utils import compute_grid_indices

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

        #import pdb;pdb.set_trace()
        irange = np.arange(iorigin, ishape, iintervals)
        jrange = np.arange(jorigin, jshape, jintervals)

        idxs_sampled = indexes[irange[:, None], jrange].flatten()  # broadcasting

        # if missing_mask is not None:
        #     idx_nan = grid_idx[missing_mask]

        #     idx_sampled_1d_nomissing = np.setdiff1d(idx_sampled, idx_nan)
        # else:
        #     idx_sampled_1d_nomissing = idx_sampled


        return idxs_sampled

# class RegularIntervalSampler(AbstractDownSampler):
#     def __init__(self, intervals: list[int], origin: list[int]):
#         self.intervals = intervals
#         self.origin = origin

#         if intervals[0] != intervals[1]:
#             raise NotImplementedError("Different x,y intervals not yet implemented!")

#         if origin[0] != origin[1]:
#             raise NotImplementedError("Different x,y origins not yet implemented!")

#     def sampling_idx(
#         self, shape, missing_mask=None, grid=None
#     ):  # remove missing is a 2D mask
#         """Sample a N-dimensional array by regularly-spaced points along the spatial axes.

#         mask_missing, removes missing values from grid where mask is True
#         """

#         xr_coords = None
#         sampled_grid = None
#         sampled_grid_dims = None

#         idx_nan = np.array([])

#         if isinstance(grid, np.ndarray):
#             shape = grid.shape
#         elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
#             shape = (len(grid.lat), len(grid.lon))
#         else:
#             pass

#         ishape, iorigin, iintervals = (
#             shape[0],
#             self.origin[0],
#             self.intervals[0],
#         )  # rows (y, lat)
#         jshape, jorigin, jintervals = (
#             shape[1],
#             self.origin[1],
#             self.intervals[1],
#         )  # columns (x, lon)

#         irange = np.arange(iorigin, ishape, iintervals)
#         jrange = np.arange(jorigin, jshape, jintervals)

#         if shape is not None:
#             grid_idx = self.compute_grid_indices(shape=shape)
#         else:
#             grid_idx = self.compute_grid_indices(grid=grid)

#         idx_sampled = grid_idx[irange[:, None], jrange].flatten()  # broadcasting

#         if missing_mask is not None:
#             idx_nan = grid_idx[missing_mask]

#             idx_sampled_1d_nomissing = np.setdiff1d(idx_sampled, idx_nan)
#         else:
#             idx_sampled_1d_nomissing = idx_sampled

#         if grid is not None:
#             if isinstance(grid, np.ndarray):
#                 sampled_grid = grid[irange[:, None], jrange]
#             elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
#                 sampled_grid = grid.isel(lat=irange, lon=jrange)

#                 xr_coords = sampled_grid.coords
#             else:
#                 pass

#             sampled_grid_dims = sampled_grid.shape  # lat, lon

#         return SamplerResult(
#             idx_grid_2d=grid_idx,
#             idx_sampled_1d=idx_sampled,
#             idx_sampled_1d_nomissing=idx_sampled_1d_nomissing,
#             idx_missing_1d=idx_nan,
#             sampled_grid=sampled_grid,
#             sampled_grid_dims=sampled_grid_dims,
#             xr_sampled_coords=xr_coords,
#         )

# class CubeletSampler(AbstractDownSampler):
#     def __init__(self):
#         pass

#     def sampling_idx(
#         self, torch_dataset
#     ):  # remove missing is a 2D mask


#         if torch_dataset.masks is not None:
#             # removed missings from indexes
#             id_sampled_1d_nomissing = torch_dataset.cbs_spatial_idxs
#         else:
#             id_sampled_1d_nomissing = None



#         return SamplerResult(
#             idx_grid_2d=None,
#             idx_sampled_1d=torch_dataset.cbs_spatial_idxs,
#             idx_sampled_1d_nomissing=id_sampled_1d_nomissing,
#             idx_missing_1d=torch_dataset.cbs_missing_idxs,
#             sampled_grid=None,
#             sampled_grid_dims=None,
#             xr_sampled_coords=None,
#         )  


# class DefaultSampler(AbstractDownSampler):
#     def __init__(self):
#         pass

#     def sampling_idx(
#         self, shape, missing_mask=None, grid=None
#     ):  # remove missing is a 2D mask

#         xr_coords = None
#         sampled_grid = None
#         sampled_grid_dims = None

#         idx_nan = np.array([])

#         if isinstance(grid, np.ndarray):
#             shape = grid.shape
#         elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
#             shape = (len(grid.lat), len(grid.lon))
#         else:
#             pass

#         ishape = shape[0]  # rows (y, lat)
#         jshape = shape[1]  # columns (x, lon)

#         irange = np.arange(0, ishape, 1)
#         jrange = np.arange(0, jshape, 1)

#         if shape is not None:
#             grid_idx = self.compute_grid_indices(shape=shape)
#         else:
#             grid_idx = self.compute_grid_indices(grid=grid)

#         idx_sampled = grid_idx[irange[:, None], jrange].flatten()  # broadcasting

#         if missing_mask is not None:
#             idx_nan = grid_idx[missing_mask]
#             idx_sampled_1d_nomissing = np.setdiff1d(idx_sampled, idx_nan)
#         else:
#             idx_sampled_1d_nomissing = idx_sampled

#         if grid is not None:
#             if isinstance(grid, np.ndarray):
#                 sampled_grid = grid[irange[:, None], jrange]
#             elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
#                 sampled_grid = grid.isel(lat=irange, lon=jrange)

#                 xr_coords = sampled_grid.coords
#             else:
#                 pass

#             sampled_grid_dims = sampled_grid.shape  # lat, lon

#         return SamplerResult(
#             idx_grid_2d=grid_idx,
#             idx_sampled_1d=idx_sampled,
#             idx_sampled_1d_nomissing=idx_sampled_1d_nomissing,
#             idx_missing_1d=idx_nan,
#             sampled_grid=sampled_grid,
#             sampled_grid_dims=sampled_grid_dims,
#             xr_sampled_coords=xr_coords,
#         )

# SAMPLERS = {
#     "downsampling_regular": RegularIntervalSampler,  
#     "default": DefaultSampler,
#     "cubelets": CubeletSampler
# }

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
        # sampling_method: str = "regular",
        # sampling_method_kwargs: dict = {},
        minibatch_sampling: str = "random",
        processing: str = "single-gpu",
    ):
    
        # self.sampling_method = sampling_method
        # self.sampling_method_kwargs = sampling_method_kwargs
        # self.method_class = SAMPLERS.get(self.sampling_method, False)

        # if not self.method_class: raise Exception(f"Available sapling methods are: {list(SAMPLERS.keys())}")

        self.minibatch_sampling = minibatch_sampling 

        self.processing = processing

    def initialize(self, torch_dataset):

        self.indexes = torch_dataset.get_indexes()

    def get_sampler(self):
        if self.processing == "single-gpu":
            if self.minibatch_sampling == "random":
                return SubsetRandomSampler(self.indexes)
            elif self.minibatch_sampling == "sequential":
                return SubsetSequentialSampler(self.indexes)
        if self.processing == "multi-gpu":
            raise NotImplementedError()

    def get_metadata(self):
        return self.result
