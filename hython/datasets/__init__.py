import torch
import xarray as xr
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
from hython.io import read_from_zarr
from hython.preprocessor import reshape

from hython.utils import (
    compute_cubelet_spatial_idxs,
    compute_cubelet_time_idxs,
    cbs_mapping_idx_slice,
    compute_cubelet_slices,
    compute_cubelet_tuple_idxs,
    compute_grid_indices,
)

try:
    import xbatcher
except:
    pass

class BaseDataset():
    pass


from .wflow import *
from .conv import *
from .rnn import *

DATASETS = {
    "LSTMDataset": LSTMDataset,
    "Wflow1d": Wflow1d,
    "Wflow1dCal": Wflow1dCal,
    "CubeletsDataset": CubeletsDataset,
}


def get_dataset(dataset):
    return DATASETS.get(dataset)


