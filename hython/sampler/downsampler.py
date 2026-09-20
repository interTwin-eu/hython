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
        seed: int | None = None
    ):
        self.frac_time = frac_time 
        self.frac_space = frac_space
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)


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




class PoolDownsampler(AbstractDownSampler):
    """Pick base cells, then take every run of each (H3).

    `RandomDownsampler` cannot do this. Run against the stacked `cell` axis it
    draws rows independently, so a base cell lands in some runs and not others,
    and the surrogate sees that cell with one parameter set instead of several
    - the failure the accumulating archive exists to prevent.

    Two things hold at once:

    **(i) Every run uses the same base cells.** The surrogate learns from same
    cell, same weather, different theta. Taking all runs of each chosen cell
    also means every run contributes the same number of rows.

    **(ii) The train/valid split never changes.** `split` partitions the pool
    once, from `split_seed`, and that partition is independent of `runs`,
    `rows_target`, `seed` and the epoch. Otherwise a cell could be in train at
    cycle 0 and in valid at cycle 3 - same weather, nearly the same parameters
    - and validation would look best exactly when the surrogate starts getting
    worse.

    `places` shrinks as runs accumulate (A2 b2: `places = rows_target // runs`),
    and the choice is a *prefix* of a fixed shuffled order, so each cycle's
    cells sit inside the previous cycle's. No cell joins training for the first
    time late in the run.

    With `resample_each_epoch`, `set_epoch` redraws which cells are used while
    keeping (i) and (ii). Validation never resamples, so its loss stays
    comparable between epochs and between cycles.
    """

    def __init__(
        self,
        pool_size: int,
        runs: int = 1,
        rows_target: int | None = None,
        seed: int | None = None,
        split: str = "train",
        valid_frac: float = 0.2,
        split_seed: int = 0,
        resample_each_epoch: bool = True,
        frac_time: float | None = None,
    ):
        if split not in ("train", "valid"):
            raise ValueError(f"split must be 'train' or 'valid', got {split!r}")
        self.pool_size = pool_size
        self.runs = runs
        self.rows_target = rows_target
        self.seed = seed
        self.split = split
        self.valid_frac = valid_frac
        self.split_seed = split_seed
        # validation is never resampled, whatever the caller asks for
        self.resample_each_epoch = resample_each_epoch and split == "train"
        self.frac_time = frac_time
        self.epoch = 0

        # Own generator, never the global one, so a draw depends on this
        # object's seed and nothing else (H6).
        self.rng = np.random.default_rng(seed)

    def base_cells(self) -> NDArray:
        """This split's base cells, in a fixed order. Never depends on epoch."""
        order = np.random.default_rng(self.split_seed).permutation(self.pool_size)
        n_valid = int(self.pool_size * self.valid_frac)
        return order[:n_valid] if self.split == "valid" else order[n_valid:]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _places(self, available: int) -> int:
        if self.rows_target is None:
            return available
        return max(1, min(available, self.rows_target // max(self.runs, 1)))

    def sampling_idx(self, coords):
        space, time = coords

        cells = self.base_cells()
        places = self._places(len(cells))

        if self.resample_each_epoch:
            # Seeded from (seed, epoch) so a run still repeats exactly.
            rng = np.random.default_rng([self.seed or 0, self.epoch])
            chosen = rng.choice(cells, size=places, replace=False)
        else:
            # A prefix of the fixed order, so a smaller `places` nests inside
            # a larger one.
            chosen = cells[:places]

        # expand each base cell to every run
        rows = np.concatenate(
            [np.sort(chosen) + m * self.pool_size for m in range(self.runs)]
        )
        rows = rows[rows < len(space)]

        if self.frac_time:
            time = np.sort(
                self.rng.choice(time, int(len(time) * self.frac_time), replace=False)
            )
        return [rows, time]
