import numpy as np
import logging
import xarray as xr
import torch
import yaml
from pathlib import Path
from copy import deepcopy
from typing import Union, Dict, Any
from omegaconf import DictConfig, OmegaConf
from dask.array import expand_dims, nanmean, nanstd, nanmin, nanmax
from hython.utils import generate_run_folder
from hython.config import Config

LOGGER = logging.getLogger(__name__)

def get_scaling_parameter(var_toscale, output_type = "numpy"):
    """Project inputs to custom range. Inputs are expected to be normalized, either
    by minmax or standard scaling"""

    # var_noscale = np.setdiff1d(var_all, list( var_toscale.keys()) )

    center = []
    scale = []

    #for var in var_all:
    for var in var_toscale.keys():
        scale.append(var_toscale[var][1] - var_toscale[var][0])
        center.append(var_toscale[var][0])

    
    if output_type == "xarray":
        #FIXME: temporary fix, need to refactor
        scale = {k:([], scale[i]) for i, k in enumerate(var_toscale.keys()) }
        center = {k:([], center[i]) for i, k in enumerate(var_toscale.keys()) }
        return  xr.Dataset(center), xr.Dataset(scale)
    else:
        return  np.array(center), np.array(scale)
    
class BaseScaler:
    def __init__(self, variable):
        self.variable = variable
        if OmegaConf.is_list(variable):
            self.variable = OmegaConf.to_container(self.variable, resolve=True)
    
    def compute(self, data, type, axes):
        """Compute the center and scale for the given data."""
        return

    @classmethod#@staticmethod
    def transform(self, data, center, scale):
        return (data - center) / scale

    def transform_inverse(self, data, center, scale):
        return (data * scale) + center
    


class BoundedScaler(BaseScaler):
    def __init__(self, variable):
        super().__init__(variable=variable)

    def compute(self, data, type, axes):
        center, scale = get_scaling_parameter(self.variable, output_type="xarray")
        return center, scale
    
class MinMax01Scaler(BaseScaler):
    def __init__(self, variable):
        super().__init__(variable)

    def compute(self, data, type, axes):
        center = data[self.variable].min(axes)
        scale = data[self.variable].max(axes) - center
        return center, scale
    

class MinMax11Scaler(BaseScaler):
    def __init__(self, variable):
        super().__init__(variable)

    def compute(self, data, type, axes):
        center = data[self.variable].min(axes)
        scale = data[self.variable].max(axes) - center
        return center, scale
    
    def transform(self, data, center, scale):
        """Transform data to the range [-1, 1]"""
        return 2 * ((data - center) / scale) - 1
    
    def transform_inverse(self, data, center, scale):
        """Inverse transform from the range [-1, 1] to the original range"""
        return ((data + 1) / 2 * scale) + center
    

class StandardScaler(BaseScaler):
    def __init__(self, variable):
        super().__init__(variable)

    def compute(self, data, type, axes):
        center = data[self.variable].mean(axes)
        scale = data[self.variable].std(axes)
        return center, scale

class Scaler2:
    """Class for performing scaling of input features. Currently supports minmax and standard scaling."""

    def __init__(
        self,
        cfg: Union[Dict, DictConfig, str],
        is_train: bool = True,
        use_cached: bool = False,
    ):
        if isinstance(cfg, Config):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            self.cfg = OmegaConf.create(cfg)
        elif isinstance(cfg, str):
            self.cfg = OmegaConf.load(cfg)
        else:
            self.cfg = cfg

        self.cfg_scaler = cfg.scaler

        self.is_train = is_train
        self.use_cached = use_cached
        self.flag_stats_computed = False

        try:
            self.run_dir = Path(generate_run_folder(cfg))
        except:
            self.run_dir = Path(".")

        LOGGER.info(f"Data statistics saved to: {str(self.run_dir)}") 

        self.archive = {}

    def set_run_dir(self, run_dir):
        self.run_dir = Path(run_dir)

    def compute2(self, data, type, axes=(0, 1)):
        scaler_list = self.cfg_scaler[type]["scaling_variant"]  
        centers, scales = [], []
        for sca in scaler_list:
            center, scale = sca.compute(data, type, axes)
            centers.append(center)
            scales.append(scale)
        # Ensure that stats has same ordering of variables listed in cfg
        center = self.ensure_var_order(xr.merge(centers), type)
        scale = self.ensure_var_order(xr.merge(scales), type)

        self.archive.update({type: {"center": center, "scale": scale}})

    def ensure_var_order(self, data, type):
        return data[list(self.cfg[type])] 

    def load_or_compute(self, data, type="dynamic_input", is_train=True, axes=(0, 1)):

        if is_train:
            if self.use_cached:
                try:
                    self.load(type)
                except FileNotFoundError:
                    LOGGER.info(f"Statistics not found in {str(self.run_dir)} for {type}, computing statistics..")            
                    self.compute2(data, type, axes)
                    self.flag_stats_computed = True
            else:
                self.compute2(data, type, axes)
                self.flag_stats_computed = True
        else:
            self.load(type)

    def transform(self, data, type):
        stats_dist = self.archive.get(type)  
        # Use subclass transform or not
        #sca = self.cfg.get(type) 
        
        if stats_dist is not None:
            #if self.cfg.scaling_variant == "minmax_11":
            #    return 2*( (data - stats_dist["center"]) / stats_dist["scale"]) -1
            return (data - stats_dist["center"]) / stats_dist["scale"]
            
    def transform_custom_range(self, data, scale, center):
        if self.cfg.scaling_variant == "minmax_11":
            return 2*( (data - center) / scale) -1
        return (data - center) / scale

    def transform_inverse_custom_range(self, data, scale, center):
        if self.cfg.scaling_variant == "minmax_11":
            return ( (data + 1)/2 * scale) + center
        return (data * scale) + center
    
    def transform_inverse(self, data, type, **kwargs):
        stats_dist = self.archive.get(type)
        scaler_list = self.cfg_scaler[type]["scaling_variant"]
        for sca in scaler_list:
            
            try: #FIXME
                # Try if dict
                var = OmegaConf.to_container(sca.variable)
            except:
                # list
                var = sca.variable

            if isinstance(var, dict):
                var = list(var.keys())

            scaled_data = sca.transform_inverse(data[var], 
                                                stats_dist["scale"][var], 
                                                stats_dist["center"][var])
            data = data.assign({v:scaled_data[v] for v in var})

        return data
        # if stats_dist is not None:
        #     if isinstance(data, xr.DataArray):
        #         stats_dist_arr = {}
        #         stats_dist_arr["center"] = stats_dist["center"].values
        #         stats_dist_arr["scale"] = stats_dist["scale"].values
        #         if self.cfg.scaling_variant == "minmax_11":
        #             return ( (data + 1)/2 * stats_dist_arr["scale"]) + stats_dist_arr["center"]
        #         return (data * stats_dist_arr["scale"]) + stats_dist_arr["center"]
        #     elif isinstance(data, xr.Dataset):
        #         if self.cfg.scaling_variant == "minmax_11":
        #             return ( (data + 1)/2 * stats_dist["scale"]) + stats_dist["center"]
        #         return (data * stats_dist["scale"]) + stats_dist["center"]
        #     else:
        #         pos = kwargs.get("var_order")
        #         if self.cfg.scaling_variant == "minmax_11":
        #             return ( (data + 1)/2 * stats_dist["scale"][pos].to_array().values) + stats_dist["center"][pos].to_array().values
        #         return (data * stats_dist["scale"][pos].to_array().values) + stats_dist["center"][pos].to_array().values

    def compute(self, data, type, axes=(0, 1)):
        "Compute assumes the features are the last dimension of the array."

        if isinstance(axes[0], int) and isinstance(data, np.ndarray):
            # workaraound for handling missing values in numpy arrays
            data = np.ma.array(data, mask=np.isnan(data))

        if "minmax" in self.cfg.scaling_variant:
            center = data.min(axes)
            scale = data.max(axes) - center
        elif self.cfg.scaling_variant == "standard":
            center = data.mean(axes)
            scale = data.std(axes)
        else:
            raise NotImplementedError(
                f"{self.cfg.scaling_variant} not yet implemented."
            )

        self.archive.update({type: {"center": center, "scale": scale}})

    def load(self, type):
        path = self.run_dir / f"{type}.yaml"
        if path.exists():
            with open(path, "r") as file:
                temp = yaml.load(file, Loader=yaml.Loader)
                try:
                    stats = {
                        type: {k: xr.Dataset.from_dict(temp[k]) for k in temp}
                    }  # loop over center, scale
                except:
                    stats = {type: {k: xr.DataArray.from_dict(temp[k]) for k in temp}}

            self.archive.update(stats)
        else:
            raise FileNotFoundError()

    def clean_cache(self, type=None):
        if type:
            path = self.run_dir / f"{type}.yaml"
            path.unlink()
        else:
            path = self.run_dir
            [f.unlink() for f in path.glob("*.yaml")]

    def write(self, type):
        stats_dict = deepcopy(self.archive.get(type))

        path = self.run_dir

        if not path.exists():
            # path may be already created when running distributed
            try: 
                path.mkdir(parents=True)
            except FileExistsError:
                pass

        # transform Dataset or DataArray to dictionary
        if isinstance(stats_dict["center"], xr.Dataset) or isinstance(
            stats_dict["center"], xr.DataArray
        ):  # loop over center, scale
            stats_dict = {k: stats_dict[k].to_dict() for k in stats_dict}

        with open(path / f"{type}.yaml", "w") as file:
            yaml.dump(stats_dict, file)


class Scaler:
    """Class for performing scaling of input features. Currently supports minmax and standard scaling."""

    def __init__(
        self,
        cfg: Union[Dict, DictConfig, str],
        is_train: bool = True,
        use_cached: bool = False,
    ):
        if isinstance(cfg, Config):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            self.cfg = OmegaConf.create(cfg)
        elif isinstance(cfg, str):
            self.cfg = OmegaConf.load(cfg)
        else:
            self.cfg = cfg

        self.is_train = is_train
        self.use_cached = use_cached
        self.flag_stats_computed = False

        try:
            self.run_dir = Path(generate_run_folder(cfg))
        except:
            self.run_dir = Path(".")

        LOGGER.info(f"Data statistics saved to: {str(self.run_dir)}") 

        self.archive = {}

    def set_run_dir(self, run_dir):
        self.run_dir = Path(run_dir)

    def load_or_compute(self, data, type="dynamic_input", is_train=True, axes=(0, 1)):
        if is_train:
            if self.use_cached:
                try:
                    self.load(type)
                except FileNotFoundError:
                    LOGGER.info(f"Statistics not found in {str(self.run_dir)} for {type}, computing statistics..")
                    self.compute(data, type, axes)
                    self.flag_stats_computed = True
            else:
                self.compute(data, type, axes)
                self.flag_stats_computed = True
        else:
            self.load(type)

    def transform(self, data, type):
        stats_dist = self.archive.get(type)

        if stats_dist is not None:
            if self.cfg.scaling_variant == "minmax_11":
                return 2*( (data - stats_dist["center"]) / stats_dist["scale"]) -1
            return (data - stats_dist["center"]) / stats_dist["scale"]

    def transform_custom_range(self, data, scale, center):
        if self.cfg.scaling_variant == "minmax_11":
            return 2*( (data - center) / scale) -1
        return (data - center) / scale

    def transform_inverse_custom_range(self, data, scale, center):
        if self.cfg.scaling_variant == "minmax_11":
            return ( (data + 1)/2 * scale) + center
        return (data * scale) + center
    
    def transform_inverse(self, data, type, **kwargs):
        stats_dist = self.archive.get(type)

        if stats_dist is not None:
            if isinstance(data, xr.DataArray):
                stats_dist_arr = {}
                stats_dist_arr["center"] = stats_dist["center"].values
                stats_dist_arr["scale"] = stats_dist["scale"].values
                if self.cfg.scaling_variant == "minmax_11":
                    return ( (data + 1)/2 * stats_dist_arr["scale"]) + stats_dist_arr["center"]
                return (data * stats_dist_arr["scale"]) + stats_dist_arr["center"]
            elif isinstance(data, xr.Dataset):
                if self.cfg.scaling_variant == "minmax_11":
                    return ( (data + 1)/2 * stats_dist["scale"]) + stats_dist["center"]
                return (data * stats_dist["scale"]) + stats_dist["center"]
            else:
                pos = kwargs.get("var_order")
                if self.cfg.scaling_variant == "minmax_11":
                    return ( (data + 1)/2 * stats_dist["scale"][pos].to_array().values) + stats_dist["center"][pos].to_array().values
                return (data * stats_dist["scale"][pos].to_array().values) + stats_dist["center"][pos].to_array().values

    def compute(self, data, type, axes=(0, 1)):
        "Compute assumes the features are the last dimension of the array."

        if isinstance(axes[0], int) and isinstance(data, np.ndarray):
            # workaraound for handling missing values in numpy arrays
            data = np.ma.array(data, mask=np.isnan(data))

        if "minmax" in self.cfg.scaling_variant:
            center = data.min(axes)
            scale = data.max(axes) - center
        elif self.cfg.scaling_variant == "standard":
            center = data.mean(axes)
            scale = data.std(axes)
        else:
            raise NotImplementedError(
                f"{self.cfg.scaling_variant} not yet implemented."
            )

        self.archive.update({type: {"center": center, "scale": scale}})

    def load(self, type):
        path = self.run_dir / f"{type}.yaml"
        if path.exists():
            with open(path, "r") as file:
                temp = yaml.load(file, Loader=yaml.Loader)
                try:
                    stats = {
                        type: {k: xr.Dataset.from_dict(temp[k]) for k in temp}
                    }  # loop over center, scale
                except:
                    stats = {type: {k: xr.DataArray.from_dict(temp[k]) for k in temp}}

            self.archive.update(stats)
        else:
            raise FileNotFoundError()

    def clean_cache(self, type=None):
        if type:
            path = self.run_dir / f"{type}.yaml"
            path.unlink()
        else:
            path = self.run_dir
            [f.unlink() for f in path.glob("*.yaml")]

    def write(self, type):
        stats_dict = deepcopy(self.archive.get(type))

        path = self.run_dir

        if not path.exists():
            # path may be already created when running distributed
            try: 
                path.mkdir(parents=True)
            except FileExistsError:
                pass

        # transform Dataset or DataArray to dictionary
        if isinstance(stats_dict["center"], xr.Dataset) or isinstance(
            stats_dict["center"], xr.DataArray
        ):  # loop over center, scale
            stats_dict = {k: stats_dict[k].to_dict() for k in stats_dict}

        with open(path / f"{type}.yaml", "w") as file:
            yaml.dump(stats_dict, file)
