import numpy as np
import xarray as xr
import torch
import yaml
from pathlib import Path
from copy import deepcopy
from typing import Union, Dict
from omegaconf import DictConfig, OmegaConf
from dask.array import expand_dims, nanmean, nanstd, nanmin, nanmax


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

