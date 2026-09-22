import os
from pathlib import Path
from timeit import default_timer
from typing import Dict, Literal, Optional, Union, Any, Tuple
from tqdm.auto import tqdm
import copy
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import pandas as pd
from ray import train

from hython.sampler import SamplerBuilder
from hython.trainer import RNNTrainer, CalTrainer
from hython.models import get_model_class as get_hython_model
from hython.models import load_model, ModelLogAPI
from itwinai.components import monitor_exec
from hython.utils import create_xarray_data

from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    TorchDDPStrategy,
)

from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.distributed import suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.inference import TorchPredictor, ModelLoader 
from itwinai.torch.type import Metric
from itwinai.torch.profiling.profiler import profile_torch_trainer

from itwinai.components import Predictor

import xarray as xr


def unscale_parameters(ds: xr.Dataset, ranges) -> xr.Dataset:
    """Physical parameter values from the parameter network's scaled output.

    The inverse of `BoundedScaler`: value = output * (upper - lower) + lower,
    with the bounds of `scaling_static_range`, the numbers the surrogate was
    trained with.
    """
    out = ds.copy()
    for p in ds.data_vars:
        lo, hi = ranges[p]
        out[p] = ds[p] * (hi - lo) + lo
    return out


class ParameterInference(Predictor):

    def __init__(self, model: str = None):
        super().__init__(model=model)
        # self.save_parameters(**self.locals2params(locals()))
        # self.model = self.model.eval()

    @monitor_exec
    def execute(
        self,
        test_dataset: Dataset,
        test_loader: Dataset,
        model: nn.Module = None,
        strategy = None,
        config = None,
        out_dir = "/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/"
    ) -> Dict[str, Any]:
        """Applies a torch model to a dataset for inference.

        Args:
            test_dataset (Dataset[str, Any]): each item in this dataset is a
                couple (item_unique_id, item)
            model (nn.Module, optional): torch model. Overrides the existing
                model, if given. Defaults to None.

        Returns:
            Dict[str, Any]: maps each item ID to the corresponding predicted
                value(s).
        """
    

        device = strategy.device()

        params = []
        model.eval()
        for data in test_loader:
            xd = data["xd"]
            xs = data["xs"]
            xp = data["xp"]
            out = model(xs.to(device), xd.to(device), xp.to(device))

            params.append(out["param"].detach())

        params = torch.concat(params, 0).detach()

        # The network's outputs are in the order of `cal_param` (see the
        # trainer, which builds TransferNN from that list).
        p = list(config.head_model_inputs["cal_param"])
        ranges = getattr(config, "scaling_static_range", None)
        if ranges is None:
            raise ValueError(
                "ParameterInference needs `scaling_static_range` in the trainer's "
                "config block: add `scaling_static_range: ${scaling_static_range}` "
                "next to `head_model_inputs`."
            )
        
        coords = xr.Coordinates({"lat":test_dataset.y.lat, "lon":test_dataset.y.lon, "variable":p})
        output_shape = {"lat":len(test_dataset.y.lat),"lon":len(test_dataset.y.lon), "variable":len(p)}
        
        ds_params = create_xarray_data(params.cpu(), 
                        coords, 
                        output_shape=output_shape
                        )

        ds_params = unscale_parameters(ds_params, ranges)


        ds_params.to_netcdf(Path(out_dir) / "inference_parameter.nc")

        return 