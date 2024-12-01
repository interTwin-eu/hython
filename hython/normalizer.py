import numpy as np
import xarray as xr
import torch
import yaml
from pathlib import Path
from copy import deepcopy
from typing import Union, Dict
from omegaconf import DictConfig, OmegaConf
from dask.array import expand_dims, nanmean, nanstd, nanmin, nanmax


FUNCS = {"minmax": [nanmin, nanmax], "standardize": [nanmean, nanstd]}

SCALER = {
    "minmax": lambda arr, axis, m1, m2: (arr - np.expand_dims(m1, axis=axis))
    / (np.expand_dims(m2, axis=axis) - np.expand_dims(m1, axis=axis)),
    "standardize": lambda arr, axis, m1, m2: (arr - expand_dims(m1, axis=axis))
    / expand_dims(m2, axis=axis),
}

SCALER_XARRAY = {
    "minmax": lambda arr, axis, m1, m2: (arr - m1) / (m2 - m1),
    "standardize": lambda arr, axis, m1, m2: (arr - m1) / m2,
}


DENORM_XARRAY = {
    "standardize": lambda arr, axis, m1, m2: (arr * m2) + m1,
    "minmax": lambda arr, axis, m1, m2: (arr * (m2 - m1)) + m1,
}

DENORM = {
    "standardize": lambda arr, axis, m1, m2: (arr * expand_dims(m2, axis=axis))
    + expand_dims(m1, axis=axis),
    "minmax": lambda arr, axis, m1, m2: (
        arr * (expand_dims(m2, axis=axis) - expand_dims(m1, axis=axis))
    )
    + expand_dims(m1, axis=axis),
}

TYPE = {
    "NTC": {"space": 0, "time": 1, "spacetime": (0, 1)},  # N T C
    "NCTHW": {"space": (1, 2), "time": 1, "spacetime": (1, 2, 3)},  # C T H W
    "NLCHW": {"space": (2, 3), "time": 0, "spacetime": (0, 2, 3)},
    "xarray_dataset": {
        "space": ("lat", "lon"),
        "time": ("time"),
        "spacetime": ("time", "lat", "lon"),
    },
}


class Normalizer:
    def __init__(
        self,
        method: str,
        type: str,
        axis_order: str,
        dask_compute: bool = False,
        save_stats: str = None,
    ):
        self.method = method
        self.axis_order = axis_order
        self.type = type
        self.dask_compute = dask_compute
        self.save_stats = save_stats

        self.stats_iscomputed = False

        self._set_axis()

    def _set_axis(self):
        axis_order = TYPE.get(self.axis_order)

        self.axis = axis_order.get(self.type, None)

    def _get_scaler(self):
        if "xarray" in self.axis_order:
            scaler = SCALER_XARRAY.get(self.method, None)
        else:
            scaler = SCALER.get(self.method, None)

        if scaler is None:
            raise NameError(f"Scaler for {self.method} does not exists")
        else:
            return scaler

    def _get_funcs(self):
        funcs = FUNCS.get(self.method, False)
        if funcs is None:
            raise NameError(f"{self.method} does not exists")
        else:
            return funcs

    def compute_stats(self, arr):
        if "xarray" in self.axis_order:
            if self.method == "standardize":
                if self.dask_compute:
                    self.computed_stats = [
                        arr.mean(self.axis).compute(),
                        arr.std(self.axis).compute(),
                    ]
                else:
                    self.computed_stats = [arr.mean(self.axis), arr.std(self.axis)]
            else:
                if self.dask_compute:
                    self.computed_stats = [
                        arr.min(self.axis).compute(),
                        arr.max(self.axis).compute(),
                    ]
                else:
                    self.computed_stats = [arr.min(self.axis), arr.max(self.axis)]

        else:
            funcs = self._get_funcs()
            self.computed_stats = [f(arr, axis=self.axis).compute() for f in funcs]

        if self.save_stats is not None and self.stats_iscomputed is False:
            self.write_stats(self.save_stats)

        self.stats_iscomputed = True

    def normalize(self, arr, read_from=None, write_to=None):
        scale_func = self._get_scaler()

        if read_from is not None:
            self.read_stats(read_from)

        if self.stats_iscomputed is False:
            self.compute_stats(arr)

        if write_to is not None:
            self.write_stats(write_to)

        if self.dask_compute:
            return scale_func(arr, self.axis, *self.computed_stats).compute()
        else:
            return scale_func(arr, self.axis, *self.computed_stats)

    def denormalize(self, arr, fp=None):
        if self.method == "standardize":
            if "xarray" in self.axis_order:
                func = DENORM_XARRAY.get(self.method)
            else:
                func = DENORM.get(self.method)

            if fp is not None:
                self.read_stats(fp)
                m, std = self.computed_stats
                return func(arr, self.axis, m, std)
            else:
                m, std = self.computed_stats
                if "xarray" in self.axis_order:
                    if isinstance(arr, np.ndarray):
                        std = std.to_dataarray().values
                        m = m.to_dataarray().values
                    else:
                        pass
                    return func(arr, self.axis, m, std)
                else:
                    return func(arr, self.axis, m, std)
        else:
            if "xarray" in self.axis_order:
                func = DENORM_XARRAY.get(self.method)
            else:
                func = DENORM.get(self.method)
            if fp is not None:
                self.read_stats(fp)
                m1, m2 = self.computed_stats
                return func(arr, self.axis, m1, m2)
            else:
                m1, m2 = self.computed_stats
                if "xarray" in self.axis_order:
                    if isinstance(arr, np.ndarray):
                        m2 = m2.to_dataarray().values
                        m1 = m1.to_dataarray().values
                    else:
                        pass
                    return func(arr, self.axis, m1, m2)
                else:
                    return func(arr, self.axis, m1, m2)

    def write_stats(self, fp):
        print(f"write stats to {fp}")
        if "xarray" in self.axis_order:
            xarr = xr.DataArray(["m1", "m2"], coords=[("stats", ["m1", "m2"])])
            ds = xr.concat(self.computed_stats, dim=xarr)
            ds.to_netcdf(fp, mode="w")
        else:
            np.save(fp, self.computed_stats)

    def read_stats(self, fp):
        print(f"read from {fp}")
        self.stats_iscomputed = True
        if "xarray" in self.axis_order:
            ds = xr.open_dataset(fp)
            ds.close()  # closing so that I can overwrite it with write stats
            self.computed_stats = [
                ds.sel(stats="m1", drop=True),
                ds.sel(stats="m2", drop=True),
            ]

        else:
            self.computed_stats = np.load(fp)

    def get_stats(self):
        return self.computed_stats


def generate_experiment_id(cfg):
    return "_".join([cfg.experiment_name, cfg.experiment_run])


class Scaler:
    """Class for performing scaling of input features. Currently supports minmax and standard scaling."""

    def __init__(
        self,
        cfg: Union[Dict, DictConfig, str],
        is_train: bool = True,
        use_cached: bool = False,
    ):
        if isinstance(cfg, dict):
            self.cfg = OmegaConf.create(cfg)
        elif isinstance(cfg, str):
            self.cfg = OmegaConf.load(cfg)
        else:
            self.cfg = cfg

        self.is_train = is_train
        self.use_cached = use_cached
        self.exp_id = generate_experiment_id(self.cfg)

        self.archive = {}

    def load_or_compute(self, data, type="dynamic_input", is_train=True, axes=(0, 1)):
        if is_train:
            if self.use_cached:
                try:
                    self.load(type)
                except FileNotFoundError:
                    self.compute(data, type, axes)
            else:
                self.compute(data, type, axes)
        else:
            self.load(type)

    def transform(self, data, type):
        stats_dist = self.archive.get(type)

        if stats_dist is not None:
            return (data - stats_dist["center"]) / stats_dist["scale"]

    def transform_inverse(self, data, type):
        stats_dist = self.archive.get(type)

        if stats_dist is not None:
            return (data * stats_dist["scale"]) + stats_dist["center"]

    def compute(self, data, type, axes=(0, 1)):
        "Compute assumes the features are the last dimension of the array."

        if isinstance(axes[0], int) and isinstance(data, np.ndarray):
            # workaraound for handling missing values in numpy arrays
            data = np.ma.array(data, mask=np.isnan(data))

        if self.cfg.scaling_variant == "minmax":
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
        path = Path(self.cfg.run_dir) / self.exp_id / f"{type}.yaml"
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
            path = Path(self.cfg.run_dir) / self.exp_id / f"{type}.yaml"
            path.unlink()
        else:
            path = Path(self.cfg.run_dir) / self.exp_id
            [f.unlink() for f in path.glob("*.yaml")]

    def write(self, type):
        stats_dict = deepcopy(self.archive.get(type))

        path = Path(self.cfg.run_dir) / self.exp_id

        if not path.exists():
            path.mkdir()

        # transform Dataset or DataArray to dictionary
        if isinstance(stats_dict["center"], xr.Dataset) or isinstance(
            stats_dict["center"], xr.DataArray
        ):  # loop over center, scale
            stats_dict = {k: stats_dict[k].to_dict() for k in stats_dict}

        with open(path / f"{type}.yaml", "w") as file:
            yaml.dump(stats_dict, file)


class SurrogateParamRescaler:
    def __init__(self, stats=None, device="cpu", type="minmax"):
        self.stats = (
            torch.tensor(stats).float().to(device) if stats is not None else None
        )
        self.type = type

    def rescale(self, param):
        if self.stats is None:
            # import pdb;pdb.set_trace()
            # minv = torch.nan_to_num(param, nan = 10e14)
            # maxv = torch.nan_to_num(param, nan = -10e14)
            return torch.sigmoid(
                param * 0.5
            )  # (param - minv.min(0)[0]) / ( maxv.max(0)[0] - minv.min(0)[0] )

        if self.type == "minmax":
            temp = self.stats[0] + torch.sigmoid(param) * (
                self.stats[1] - self.stats[0]
            )
            return (temp - self.stats[0]) / (self.stats[1] - self.stats[0])
        elif self.type == "standardize":
            return param * self.stats[1] + self.stats[0]
