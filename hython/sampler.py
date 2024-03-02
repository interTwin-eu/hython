import torch 
import numpy as np
import xarray as xr

from abc import ABC, abstractmethod
from dataclasses import dataclass


# type hints
from typing import Any, Tuple, List
from numpy.typing import NDArray
from xarray.core.coordinates import DatasetCoordinates

from torch.utils.data import Sampler

class SpaceSampler(Sampler):
    
    def __init__(self, data_source, num_samples= 10, generator = None, sampling_indices = None):
        
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.sampling_indices = sampling_indices
        
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        yield from self.sampling_indices
    
    def __len__(self) -> int:
        return self.num_samples


@dataclass
class SamplerResult:
    """Store metadata to restructure original grid from the sampled grid
    """
    idx_grid_2d: NDArray
    idx_sampled_1d: NDArray 
    idx_sampled_1d_nomissing: NDArray | None
    idx_missing_1d: NDArray | None
    sampled_grid: NDArray | None
    sampled_grid_dims: tuple | None
    xr_sampled_coords: DatasetCoordinates | None

    def __repr__(self):
        return f'SamplerResult(\n - id_grid_2d: {self.idx_grid_2d.shape} \n - idx_sampled_1d: {self.idx_sampled_1d.shape} \n - idx_sampled_1d_nomissing: {self.idx_sampled_1d_nomissing.shape}) \n - idx_missing_1d: {self.idx_missing_1d.shape} \n - sampled_grid_dims: {self.sampled_grid_dims} \n - xr_coords: {self.xr_sampled_coords}'


class AbstractSampler(ABC):
    
    def __init__(self):
        """Pass parametes required by the sampling approach
        """    
        pass

    # def __post_init__(self):
    #     self._has_required_attributes()

    # def _has_required_attributes(self):
    #     req_attrs: List[str] = ['grid']
    #     for attr in req_attrs:
    #         if not hasattr(self, attr):
    #             raise AttributeError(f"Missing attribute: '{attr}'")
    
    @abstractmethod
    def sampling_idx(self, grid: NDArray | xr.DataArray | xr.Dataset) -> SamplerResult:
        """Sample the original grid. Must be instantiated by a concrete class that implements the sampling approach.

        Args:
            grid (NDArray | xr.DataArray | xr.Dataset): The gridded data to be sampled

        Returns:
            Tuple[NDArray, SamplerMetaData]: The sampled grid and sampler's metadata
        """
        
        pass
    
        

class RegularIntervalSampler(AbstractSampler):

    def __init__(self,
                intervals: tuple[int] = (5,5), 
                origin: tuple[int] = (0, 0)):
        
        self.intervals = intervals
        self.origin = origin

        if intervals[0] != intervals[1]:
            raise NotImplementedError("Different x,y intervals not yet implemented!")

        if origin[0] != origin[1]:
            raise NotImplementedError("Different x,y origins not yet implemented!")

    def sampling_idx(self, grid, missing_mask = None, return_grid = False): # remove missing is a 2D mask
        
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
        
        if isinstance(grid, np.ndarray):
            shape = grid.shape
        elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
            shape = (len(grid.lat), len(grid.lon))
        else:
            pass
        
        ishape,iorigin,iintervals = shape[0], self.origin[0], self.intervals[0] # rows (y, lat)
        jshape,jorigin,jintervals = shape[1], self.origin[1], self.intervals[1] # columns (x, lon)

        irange = np.arange(iorigin, ishape, iintervals)
        jrange = np.arange(jorigin, jshape, jintervals)

        grid_idx = np.arange(0, ishape * jshape, 1).reshape(ishape, jshape)

        idx_sampled = grid_idx[irange[:,None], jrange].flatten() # broadcasting
        
        if missing_mask is not None:
            idx_nan = grid_idx[missing_mask]
            
            idx_sampled_1d_nomissing = np.setdiff1d(idx_sampled, idx_nan)
        else: 
            idx_sampled_1d_nomissing = missing_mask


        if isinstance(grid, np.ndarray):
            sampled_grid = grid[irange[:, None], jrange]
        elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
            
            sampled_grid = grid.isel(lat=irange, lon=jrange)

            xr_coords = sampled_grid.isel(variable=0, drop=True).coords
            #xr_coords.assign({"lat":xr_coords["lat"][irange]})
            #xr_coords.assign({"lon":xr_coords["lon"][jrange]})
            
        else:
            pass
        
        sampled_grid_dims = sampled_grid.shape # lat, lon
        
                
        return SamplerResult(   idx_grid_2d = grid_idx, 
                                idx_sampled_1d = idx_sampled, 
                                idx_sampled_1d_nomissing = idx_sampled_1d_nomissing,
                                idx_missing_1d = idx_nan,
                                sampled_grid = sampled_grid,
                                sampled_grid_dims = sampled_grid_dims,
                                xr_sampled_coords = xr_coords )



class StratifiedSampler(AbstractSampler):
    pass



class SpatialCorrSampler(AbstractSampler):
    pass

