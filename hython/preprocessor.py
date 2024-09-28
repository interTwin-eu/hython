import xarray as xr
import numpy as np
from typing import List
from dask.array.core import Array as DaskArray


def reshape(
    data, type = "dynamic", return_type= "xarray"
):
    if type == "dynamic":
        D = (
            data.to_dataarray(dim="feat") # cast
            .stack(gridcell=["lat", "lon"])  # stack
            .transpose("gridcell", "time", "feat")
            )
        print("dynamic: ", D.shape, " => (GRIDCELL, TIME, FEATURE)")
    elif type == "static":
        D = (
            data.drop_vars("spatial_ref")
            .to_dataarray(dim="feat")
            .stack(gridcell=["lat", "lon"])
            .transpose("gridcell", "feat")
            )
        print("static: ", D.shape, " => (GRIDCELL, FEATURE)")  
    elif type == "target":

        D = (
            data.to_dataarray(dim="feat")
            .stack(gridcell=["lat", "lon"])
            .transpose("gridcell", "time", "feat")
        )        
        print("target: ", D.shape, " => (GRIDCELL, TIME, FEATURE)")

    if return_type == "xarray":
        return D
    if return_type == "dask":
        return D.data
    if return_type == "numpy":
        return D.compute().values


