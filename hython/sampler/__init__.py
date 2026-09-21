import numpy as np
import xarray as xr

from torch.utils.data import Dataset
from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import (
    SubsetRandomSampler,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
)


# == TEMPORAL DOWNSAMPLING (H8)
#
# The dataset holds every start day; these samplers pick which ones the loader
# visits. Training draws a new subset every epoch, validation draws one subset
# once and keeps it, so its loss only moves when the model does - early
# stopping, the learning-rate scheduler and the multicycle loop all read it.
#
# Each sampler owns a generator seeded from `dynamic_downsampler.seed`. The
# earlier code seeded the *global* numpy state (changing it for everything
# else in the process) and then drew with Python's `random`, which that seed
# never touched, so no draw could be repeated.


def _n_start_days(data_source) -> int:
    return data_source.time_size - data_source.seq_len


def _n_cells(data_source) -> int:
    return len(data_source.cell_coords[data_source.cell_linear_index])


def _subset_size(n_start: int, frac) -> int:
    """`frac` of the start days; None or >= 1 means all of them."""
    if frac is None or frac >= 1:
        return n_start
    return int(n_start * frac)


def _draw_days(rng, n_start: int, size: int, replacement: bool) -> np.ndarray:
    if size >= n_start and not replacement:
        return np.arange(n_start)
    if replacement:
        return rng.integers(0, n_start, size)
    return rng.choice(n_start, size, replace=False)


def _flat_index(days: np.ndarray, n_start: int, n_cells: int) -> np.ndarray:
    """Dataset indices for `days` of every cell, cell-major.

    Same order as `hython.utils.generate_time_idx`: all chosen days of cell 0,
    then of cell 1, and so on - index `c * n_start + day`, which is where
    `spacetime_index` keeps (cell c, day).
    """
    return (np.arange(n_cells)[:, None] * n_start + np.asarray(days)[None, :]).ravel()


def _valid_frac(dynamic_downsampler):
    """`frac_time_valid` if the config sets it, else the shared `frac_time`."""
    if "frac_time_valid" in dynamic_downsampler:
        return dynamic_downsampler.get("frac_time_valid")
    return dynamic_downsampler.get("frac_time")


class _TemporalDraw:
    """What the three samplers below share.

    `fixed=True` draws the days once, here, and returns the same sorted indices
    every epoch (validation). `fixed=False` draws new days at every `__iter__`
    and shuffles the result across cells (training), from the same generator,
    so a run still repeats exactly.
    """

    def _setup(self, data_source, dynamic_downsampler, replacement, fixed):
        self.data_source = data_source
        self.replacement = replacement
        self.fixed = fixed
        self.seq_len = data_source.seq_len
        self.time_size = data_source.time_size
        self.cell_size = _n_cells(data_source)
        self.seed = dynamic_downsampler.get("seed")
        self.rng = np.random.default_rng(self.seed)

        frac = _valid_frac(dynamic_downsampler) if fixed else dynamic_downsampler.get("frac_time")
        self.n_start = _n_start_days(data_source)
        self.temporal_subset_size = _subset_size(self.n_start, frac)
        self.total_subset_size = self.temporal_subset_size * self.cell_size

        self.time_indices = None
        if fixed:
            self.time_indices = np.sort(self._draw())
            self._fixed_index = _flat_index(self.time_indices, self.n_start, self.cell_size)

    def _draw(self) -> np.ndarray:
        return _draw_days(self.rng, self.n_start, self.temporal_subset_size, self.replacement)

    def _indices(self) -> np.ndarray:
        if self.fixed:
            return self._fixed_index
        # Shuffled across cells and days. The loader keeps a sampler's order
        # as given, and the flat index is cell-major, so without this a batch
        # of 512 held only ~6 cells (H8).
        self.time_indices = self._draw()
        return self.rng.permutation(_flat_index(self.time_indices, self.n_start, self.cell_size))

    def __iter__(self):
        return iter(self._indices().tolist())

    def __len__(self):
        return self.total_subset_size


class RandomTemporalDynamicDownsampler(_TemporalDraw, RandomSampler):
    """Training: a new random subset of start days every epoch, the same for
    every cell, visited in random order across cells. `frac_time` sets its
    size."""

    def __init__(self, data_source, dynamic_downsampler, replacement=False):
        RandomSampler.__init__(self, data_source)
        self._setup(data_source, dynamic_downsampler, replacement, fixed=False)


class SequentialTemporalDynamicDownsampler(_TemporalDraw, RandomSampler):
    """Validation: one subset of start days, drawn once and kept for every
    epoch, visited in order. `frac_time_valid` sets its size (None = every
    start day); without it, the shared `frac_time`."""

    def __init__(self, data_source, dynamic_downsampler, replacement=False):
        RandomSampler.__init__(self, data_source)
        self._setup(data_source, dynamic_downsampler, replacement, fixed=True)


class DistributedTemporalDynamicDownsampler(_TemporalDraw, DistributedSampler):
    """The two above under a distributed strategy: `shuffle=True` behaves as
    training, `shuffle=False` as validation.

    Not checked: `__iter__` does not split the indices by rank, so every
    worker visits every sample. Left for later - runs are on one GPU (H8).
    """

    def __init__(self, data_source, dynamic_downsampler, shuffle=True,
                 replacement=False, **sampling_kwargs):
        DistributedSampler.__init__(self, dataset=data_source, shuffle=shuffle,
                                    **sampling_kwargs)
        self._setup(data_source, dynamic_downsampler, replacement, fixed=not shuffle)


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
