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
            
            if "zscore" in self.method:
                kind = self.kwargs.get("kind")
                if kind == "local":
                    self.reference_scale = vs.std("time").compute()
                    self.reference_center = vs.mean("time").compute()
                    self.target_scale = data[self.variable].std("time")
                    self.target_center = data[self.variable].mean("time")
                elif kind == "global":
                    self.reference_scale = vs.std().compute()
                    self.reference_center = vs.mean().compute()
                    self.target_scale = data[self.variable].std()
                    self.target_center = data[self.variable].mean()
                else:
                    raise NotImplementedError
                
            elif self.method == "minmax":
                self.reference_center = vs.min("time").compute()
                self.reference_scale = vs.max("time").compute() - self.reference_center
                self.target_center = data[self.variable].min("time")
                self.target_scale = data[self.variable].max("time") - self.target_center
            elif self.method == "iqr": # wflow outliers
                self.reference_center = vs.min("time").compute()
                vs = vs.chunk(dict(time=-1)).load()
                self.reference_scale = vs.quantile(dim= "time", q=0.9) - self.reference_center
                self.target_center = data[self.variable].min("time")
                self.target_scale = data[self.variable].max("time") - self.target_center
                
        return self.reference_scale, self.reference_center

    def transform(self, data):
        if self.method == "zscore-nonneg":
            return np.maximum((self.reference_scale / self.target_scale) * (data - self.target_center) + self.reference_center, 0)
        else:
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
                # `parents=True`: a work_dir whose parent does not exist used to
                # raise here and fall through to the working directory below,
                # silently writing statistics wherever the process started.
                self.run_dir.mkdir(parents=True, exist_ok=True)
        except Exception as err:
            # Falling back to the working directory writes statistics wherever
            # the process happens to be started, which silently overwrites any
            # `<type>.yaml` already sitting there. Kept for compatibility, but
            # no longer silent: a config without `work_dir` used to corrupt the
            # cached numbers with no visible sign.
            self.run_dir = Path(".")
            LOGGER.warning(
                f"Could not resolve the run folder ({err!r}); falling back to "
                f"{self.run_dir.resolve()}. Statistics written there may "
                f"overwrite unrelated files - set `work_dir` to avoid this."
            )
            
        # with open(self.run_dir / f"config.yaml", "w") as file:
        #     import pdb;pdb.set_trace()
        #     y = OmegaConf.to_yaml(self.cfg)
        #     yaml.dump(y, file)
        
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

    def ensure_var_order(self, data, type1):
        from typing import Mapping

        if isinstance(self.cfg[type1], Mapping):
            head_model_input_list = []
            for i in self.cfg[type1]:
                if i is not None and i != "cal_param":
                    head_model_input_list.extend(self.cfg[type1][i])
            return data[head_model_input_list] 
        else:
            return data[list(self.cfg[type1])] 

    def load_or_compute(self, data, type="dynamic_inputs", is_train=True, axes=(0, 1), **kwargs):
        """Load cached statistics, or compute them, for one variable group.

        `use_cached` is strict (H4). It used to fall back to recomputing when
        the cache was missing, which on the pool archive would quietly derive
        the numbers from 5% of the map and hand calibration a different scale
        than training - with no error anywhere. If the cache is asked for, it
        has to be there.
        """
        if is_train:
            if self.use_cached:
                self.load(type)
            else:
                self.compute(data, type, axes, **kwargs)
                self.flag_stats_computed = True
        else:
            self.load(type)

        self.apply_frozen(type)

    def apply_frozen(self, type):
        """Override named variables with statistics frozen on the full map (H4).

        Five statics - `wflow_uparea`, `wflow_landuse`, `wflow_dem`, `Slope`,
        `WaterFrac` - are MinMax01-scaled in *both* configs: as part of
        `static_inputs` when training the surrogate, and as
        `head_model_inputs.aux_feat` when calibrating. Each side works its own
        numbers out, and after A1 the training side only sees the 5% pool,
        whose maxima are far below the map's (`WaterFrac` reached 0.567 against
        a true 0.905). The same cell would then mean two different numbers on
        the two sides, with nothing raising.

        Matching is by variable name, so one frozen file serves both groups
        even though they sit in different `type`s and different files.
        """
        # `getattr` rather than `.get`: cfg may be a DictConfig or a Config,
        # and OmegaConf's missing-key error subclasses AttributeError, so this
        # form works for both.
        path = getattr(self.cfg, "scaling_frozen_stats", None)
        if not path:
            return

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"`scaling_frozen_stats` points at {path}, which does not "
                f"exist. Write it once from the full map before training."
            )

        stats = self.archive.get(type)
        if stats is None:
            return

        with open(path, "r") as file:
            temp = yaml.load(file, Loader=yaml.Loader)
        frozen = {k: xr.Dataset.from_dict(temp[k]) for k in temp}

        shared = [v for v in frozen["center"].data_vars if v in stats["center"].data_vars]
        if not shared:
            return

        for k in ("center", "scale"):
            stats[k] = stats[k].assign({v: frozen[k][v] for v in shared})
        self.archive[type] = stats
        LOGGER.info(f"{type}: {len(shared)} variable(s) frozen from {path}: {shared}")

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

            if subset:
                return data

        return data

    def transform_inverse_custom_range(self, data, scale, center):
        return (data * scale) + center
    

    def load(self, type):
        """Read cached statistics for one variable group.

        Raises if they are not there. It used to return silently, which left
        `self.archive` without the group and pushed the failure somewhere far
        from the cause. Groups with no scaler configured (`target_variables:
        null`) have nothing to read and are skipped.
        """
        if self.cfg_scaler.get(type) is None:
            return

        path = self.run_dir / f"{type}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached statistics for '{type}' at {path}. "
                f"They are written by the training run that computes them; "
                f"with `scaling_use_cached: true` they must already exist."
            )
        with open(path, "r") as file:
            temp = yaml.load(file, Loader=yaml.Loader)
            try:
                stats = {
                    type: {k: xr.Dataset.from_dict(temp[k]) for k in temp}
                }  # loop over center, scale
            except Exception:
                stats = {type: {k: xr.DataArray.from_dict(temp[k]) for k in temp}}

        self.archive.update(stats)

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


