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


# def reshape(
#     dynamic: xr.Dataset, static: xr.Dataset, target: xr.Dataset, return_type="xarray"
# ) -> List[DaskArray] | List[xr.DataArray]:
#     # reshape
#     Xd = (
#         dynamic.to_dataarray(dim="feat").astyoe(np.float16)  # cast
#         .stack(gridcell=["lat", "lon"])  # stack
#         .transpose("gridcell", "time", "feat")
#     )
#     print("dynamic: ", Xd.shape, " => (GRIDCELL, TIME, FEATURE)")

#     Xs = (
#         static.drop_vars("spatial_ref")
#         .to_dataarray(dim="feat").astyoe(np.float16) 
#         .stack(gridcell=["lat", "lon"])
#         .transpose("gridcell", "feat")
#     )
#     print("static: ", Xs.shape, " => (GRIDCELL, FEATURE)")

#     Y = (
#         target.to_dataarray(dim="feat").astyoe(np.float16) 
#         .stack(gridcell=["lat", "lon"])
#         .transpose("gridcell", "time", "feat")
#     )
#     print("target: ", Y.shape, " => (GRIDCELL, TIME, TARGET)")


#     if return_type == "xarray":
#         return Xd, Xs, Y
#     if return_type == "dask":
#         return Xd.data, Xs.data, Y.data
#     if return_type == "numpy":
#         return Xd.compute().values, Xs.compute().values, Y.compute().values




