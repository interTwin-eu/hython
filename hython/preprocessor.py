import xarray as xr
import numpy as np
from typing import List
from omegaconf import DictConfig, OmegaConf
from dask.array.core import Array as DaskArray


class BasePreprocessor:
    def __init__(self, variable):
        self.variable = variable
        if OmegaConf.is_list(variable):
            self.variable = OmegaConf.to_container(self.variable, resolve=True)


class Log(BasePreprocessor):
    def __init__(self, variable: list[str]):
        self.variable = variable

    def __call__(self, data: xr.Dataset) -> xr.Dataset:
        return data.assign({v:np.log1p(data[v]) for v in self.variable})


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg.preprocessor 

    def preprocess(self, data, type):
        if self.cfg.get(type) is None:
            return data
        prepr_list = self.cfg[type]["variant"]
        for pre in prepr_list:
            try: #FIXME
                # Try if dict
                var = OmegaConf.to_container(pre.variable)
            except:
                # list
                var = pre.variable

            if isinstance(var, dict):
                var = list(var.keys())

            prepr_data = pre(data[var])

            return data.assign({v:prepr_data[v] for v in var})

        
def reshape(data, type="dynamic", return_type="xarray"):
    if type == "dynamic":
        D = (
            data.to_dataarray(dim="feat")  # cast
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


