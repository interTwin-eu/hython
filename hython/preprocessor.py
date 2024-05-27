import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import xarray as xr

from typing import List
from typing import Union, Any
from dask.array.core import Array as DaskArray

from dask.array import expand_dims, nanmean, nanstd, nanmin, nanmax


def reshape(
    dynamic: xr.Dataset, static: xr.Dataset, target: xr.Dataset, return_type="xarray"
) -> List[DaskArray] | List[xr.DataArray]:
    # reshape
    Xd = (
        dynamic.to_dataarray(dim="feat")  # cast
        .stack(gridcell=["lat", "lon"])  # stack
        .transpose("gridcell", "time", "feat")
    )
    print("dynamic: ", Xd.shape, " => (GRIDCELL, TIME, FEATURE)")

    Xs = (
        static.drop_vars("spatial_ref")
        .to_dataarray(dim="feat")
        .stack(gridcell=["lat", "lon"])
        .transpose("gridcell", "feat")
    )
    print("static: ", Xs.shape, " => (GRIDCELL, FEATURE)")

    Y = (
        target.to_dataarray(dim="feat")
        .stack(gridcell=["lat", "lon"])
        .transpose("gridcell", "time", "feat")
    )
    print("target: ", Y.shape, " => (GRIDCELL, TIME, TARGET)")

    if return_type == "xarray":
        return Xd, Xs, Y
    if return_type == "dask":
        return Xd.data, Xs.data, Y.data
    if return_type == "numpy":
        return Xd.compute().values, Xs.compute().values, Y.compute().values
