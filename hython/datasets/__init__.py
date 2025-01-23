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
    generate_run_folder,
    get_source_url
)

try:
    import xbatcher
except:
    pass


class BaseDataset(Dataset):
    def get_scaling_parameter(self, var_toscale, var_all):
        """Project inputs to custom range. Inputs are expected to be normalized, either
        by minmax or standard scaling"""

        # var_noscale = np.setdiff1d(var_all, list( var_toscale.keys()) )

        center = []
        scale = []

        for var in var_all:
            if var in var_toscale.keys():
                scale.append(var_toscale[var][1] - var_toscale[var][0])
                center.append(var_toscale[var][0])
            else:
                scale.append(1)
                center.append(0)

        return np.array(scale), np.array(center)


from .wflow_sbm import *

DATASETS = {
    "Wflow1d": Wflow1d,
    "Wflow1dCal": Wflow1dCal,
    "Wflow2d": Wflow2d,
    "Wflow2dCal": Wflow2dCal,
}


def get_dataset(dataset):
    return DATASETS.get(dataset)
