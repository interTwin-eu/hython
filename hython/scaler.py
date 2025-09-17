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
    def __init__(self, variable, **kwargs):
        self.variable = variable
        self.kwargs = kwargs
        if OmegaConf.is_list(variable):
            self.variable = OmegaConf.to_container(self.variable, resolve=True)
    
    def compute(self, data, type, axes, **kwargs):
        """Compute the center and scale for the given data."""
        raise NotImplementedError()

    @classmethod
    def transform(self, data, center, scale):
        return (data - center) / scale

    def transform_inverse(self, data, center, scale):
        return (data * scale) + center
    
    def update_attribute(self, data):
        up = {"variant":self.__class__.__name__}
        for v in self.variable:
            data[v].attrs.update(up)
    
class BoundedScaler(BaseScaler):
    def __init__(self, variable):
        super().__init__(variable=variable)

    def compute(self, data, type, axes, **kwargs):
        center, scale = get_scaling_parameter(self.variable, output_type="xarray")
        self.update_attribute(center)
        self.update_attribute(scale)
        return center, scale
    
class MinMax01Scaler(BaseScaler):
    def __init__(self, variable):
        super().__init__(variable)

    def compute(self, data, type, axes, **kwargs):
        center = data[self.variable].min(axes)
        scale = data[self.variable].max(axes) - center
        self.update_attribute(center)
        self.update_attribute(scale)
        return center, scale
    
class MinMax11Scaler(BaseScaler):
    def __init__(self, variable):
        super().__init__(variable)

    def compute(self, data, type, axes, **kwargs):
        center = data[self.variable].min(axes)
        scale = data[self.variable].max(axes) - center
        self.update_attribute(center)
        self.update_attribute(scale)
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

    def compute(self, data, type, axes, **kwargs):
        center = data[self.variable].mean(axes)
        scale = data[self.variable].std(axes)
        self.update_attribute(center)
        self.update_attribute(scale)
        return center, scale

class TargetCalibrationScaler(BaseScaler):
    def __init__(self, variable, **kwargs):
        super().__init__(variable, **kwargs.get("kwargs"))
    
    def compute(self, data, type, axes, reference, period_range):
        print(period_range)
        how = self.kwargs.get("how")
        ref2target_mapping = self.kwargs.get("ref2target")
        self.method = self.kwargs.get("method")     
        ref_var = list(ref2target_mapping.keys())
        
        if how == "soil-property":
            pass
            #lower = how["lower"]
            #upper = how["upper"]
            # par = read_from_zarr(url=urls["static_parameter_inputs"], chunks="auto")[[lower, upper]]
            # self.y = super().rescale_target(self.y, par[lower], par[upper])
        elif how == "reference-statistics":

            vs = reference[ref_var].sel(time=period_range)
            vs = vs.rename_vars(ref2target_mapping)
            
            if self.method == "zscore":
                kind = self.kwargs.get("kind")
                if kind == "local":
                    self.reference_scale = vs.std("time").compute()
                    self.reference_center = vs.mean("time").compute()
                    self.target_scale = data[self.variable].std("time")
                    self.target_center = data[self.variable].mean("time")
                elif kind == "global":
                    self.reference_scale = vs.std()
                    self.reference_center = vs.mean()
                    self.target_scale = data[self.variable].std()
                    self.target_center = data[self.variable].mean()
                else:
                    raise
            elif self.method == "minmax":
                self.reference_center = vs.min("time")
                self.reference_scale = vs.max("time") - self.reference_center
                self.target_center = data[self.variable].min("time")
                self.target_scale = data[self.variable].max("time") - self.target_center

        return self.reference_scale, self.reference_center

    def transform(self, data):
        return (self.reference_scale / self.target_scale) * (data - self.target_center) + self.reference_center
    

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
        
        self.cfg_scaler = cfg.scaler

        self.is_train = is_train
        self.use_cached = use_cached
        self.flag_stats_computed = False

        try:
            self.run_dir = Path(generate_run_folder(cfg))
            if not self.run_dir.exists():
                self.run_dir.mkdir()
        except:
            self.run_dir = Path(".")
            
        with open(self.run_dir / f"config.yaml", "w") as file:
            yaml.dump(self.cfg, file)
        
        LOGGER.info(f"Data statistics saved to: {str(self.run_dir)}") 

        self.archive = {}

    def set_run_dir(self, run_dir):
        self.run_dir = Path(run_dir)

    def compute(self, data, type, axes=(0, 1), **kwargs):

        if self.cfg_scaler[type] is not None:  
            scaler_list = self.cfg_scaler[type]["variant"]
        else:
            return
        
        centers, scales = [], []
        for sca in scaler_list:
            center, scale = sca.compute(data, type, axes, **kwargs)
            centers.append(center)
            scales.append(scale)

        if isinstance(sca, TargetCalibrationScaler):
            # FIXME: the TargetCalibrationScaler class should return both target and reference's center and scale
            # this requires a refactoring of the compute logic
            return 

        # Ensure that stats has same ordering of variables listed in cfg
        center = self.ensure_var_order(xr.merge(centers), type)
        scale = self.ensure_var_order(xr.merge(scales), type)

        self.archive.update({type: {"center": center, "scale": scale}})

    def ensure_var_order(self, data, type):
        if isinstance(self.cfg[type],dict):
            head_model_input_list = []
            for i in self.cfg[type]:
                if i is not None and i != "cal_param":
                    head_model_input_list.extend(self.cfg[type][i])
            return data[head_model_input_list] 
        else:
            return data[list(self.cfg[type])] 

    def load_or_compute(self, data, type="dynamic_inputs", is_train=True, axes=(0, 1), **kwargs):
        if is_train:
            if self.use_cached:
                try:
                    self.load(type)
                except FileNotFoundError:
                    LOGGER.info(f"Statistics not found in {str(self.run_dir)} for {type}, computing statistics..")            
                    self.compute(data, type, axes, **kwargs)
                    self.flag_stats_computed = True
            else:
                self.compute(data, type, axes, **kwargs)
                self.flag_stats_computed = True
        else:
            self.load(type)

    def transform(self, data, type):
        stats_dist = self.archive.get(type)

        if self.cfg_scaler[type] is not None:  
            scaler_list = self.cfg_scaler[type]["variant"]
        else:
            return data
        
        for sca in scaler_list:
            
            try: #FIXME
                # Try if dict
                var = OmegaConf.to_container(sca.variable)
            except:
                # list
                var = sca.variable

            if isinstance(var, dict):
                var = list(var.keys())

            if isinstance(sca, TargetCalibrationScaler):
                scaled_data = sca.transform(data[var])
                data = data.assign({v:scaled_data[v] for v in var})
            else:
                scaled_data = sca.transform(data[var], 
                                            stats_dist["center"][var],
                                            stats_dist["scale"][var])
                data = data.assign({v:scaled_data[v] for v in var})

        return data
    
    def transform_inverse(self, data, type, **kwargs):
        stats_dist = self.archive.get(type)

        if self.cfg_scaler[type] is not None:  
            scaler_list = self.cfg_scaler[type]["variant"]
        else:
            return data
        
        for sca in scaler_list:
            
            try: #FIXME
                # Try if dict
                var = OmegaConf.to_container(sca.variable)
            except:
                # list
                var = sca.variable
 
            if isinstance(var, dict):
                var = list(var.keys())

            if subset := kwargs.get("subset", False):
                # var is a list of variables to be scaled and it is 
                # defined in the scaler keyword of the config.yaml.
                # In calibration it can happen that the scaler from the
                # training config have more input variables.
                var = subset

            unscaled_data = sca.transform_inverse(data[var], 
                                                stats_dist["center"][var],
                                                stats_dist["scale"][var])
            data = data.assign({v:unscaled_data[v] for v in var})

        return data

    def transform_inverse_custom_range(self, data, scale, center):
        return (data * scale) + center
    

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
        # else:
        #     raise FileNotFoundError()

    def clean_cache(self, type=None):
        if type:
            path = self.run_dir / f"{type}.yaml"
            path.unlink()
        else:
            path = self.run_dir
            [f.unlink() for f in path.glob("*.yaml")]

    def write(self, type):
        stats_dict = deepcopy(self.archive.get(type))

        if stats_dict is None:
            return

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


