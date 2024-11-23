import torch
import xarray as xr
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

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



from .conv import *
from .rnn import *

DATASETS = {
    "LSTMDataset": LSTMDataset,
    "XBatchDataset": XBatchDataset,
    "CubeletsDataset": CubeletsDataset,
}


def get_dataset(dataset):
    return DATASETS.get(dataset)


