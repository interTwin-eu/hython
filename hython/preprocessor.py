import xarray as xr
import numpy as np
from omegaconf import OmegaConf
import dask.array as dk


class BasePreprocessor:
    def __init__(self, variable):
        self.variable = variable
        if OmegaConf.is_list(variable):
            self.variable = OmegaConf.to_container(self.variable, resolve=True)

    def process(self, data: xr.Dataset) -> xr.Dataset:
        return NotImplementedError()
    
    def process_inverse(self, data: xr.Dataset) -> xr.Dataset:
        return NotImplementedError()


class Log1p(BasePreprocessor):
    def __init__(self, variable: list[str]):
        self.variable = variable

    def process(self, data: xr.Dataset, lazy = False) -> xr.Dataset:
        func = dk.log1p if lazy else np.log1p 
        return data.assign({v: func(data[v]) for v in self.variable})
    
    def process_inverse(self, data: xr.Dataset, lazy=False) -> xr.Dataset:
        func = dk.exp if lazy else np.exp 
        return func(data) - 1 
    

class Log10p(BasePreprocessor):
    def __init__(self, variable: list[str]):
        self.variable = variable

    def process(self, data: xr.Dataset, lazy = False) -> xr.Dataset:

        func = dk.log10 if lazy else np.log10 

        func2 = lambda x: func(x + 1)

        return data.assign({v:func2(data[v]) for v in self.variable})
    
    def process_inverse(self, data: xr.Dataset, lazy = False) -> xr.Dataset:
        #func = dk.exp if lazy else np.exp 
        return (10**data) - 1 


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg.preprocessor 

    def process(self, data, type):
        if self.cfg.get(type) is None:
            print("Type not defined, returning data unchanged.")
            return data
        prepr_list = self.cfg[type]["variant"]
        lazy = False if self.cfg[type].get("lazy") is None else self.cfg[type].get("lazy")

        for preprocessor in prepr_list:
            try: #FIXME
                # Try if dict
                var = OmegaConf.to_container(preprocessor.variable)
            except:
                # list
                var = preprocessor.variable
            
            if isinstance(var, dict):
                var = list(var.keys())

            prepr_data = preprocessor.process(data[var], lazy = lazy)

            return data.assign({v:prepr_data[v] for v in var})


    def process_inverse(self, data, type):
        if self.cfg.get(type) is None:
            return data
        prepr_list = self.cfg[type]["variant"]
        lazy = False if self.cfg[type].get("lazy") is None else self.cfg[type].get("lazy")

        for preprocessor in prepr_list:
            try: #FIXME
                # Try if dict
                var = OmegaConf.to_container(preprocessor.variable)
            except:
                # list
                var = preprocessor.variable
            
            if isinstance(var, dict):
                var = list(var.keys())

            prepr_data = preprocessor.process_inverse(data[var], lazy = lazy)

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


