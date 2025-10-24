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

from hython.scaler import Scaler

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
from omegaconf import OmegaConf
from hydra.utils import instantiate

import xarray as xr

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
        config = None
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

        p = ["KsatVer", "c", "f", "RootingDepth", "Sl"]
        
        coords = xr.Coordinates({"lat":test_dataset.y.lat, "lon":test_dataset.y.lon, "variable":p})
        output_shape = {"lat":len(test_dataset.y.lat),"lon":len(test_dataset.y.lon), "variable":len(p)}
        
        ds_params = create_xarray_data(params.cpu(), 
                        coords, 
                        output_shape=output_shape
                        )


        cfg_head_layer=OmegaConf.load("/home/iferrario/dev/article/param_estimation/config/config_training_calibration_loop.yaml")
        cfg_head_layer.pop("training_pipeline")
        cfg_head_layer = instantiate(cfg_head_layer)

        scaler_head_layer = Scaler(cfg_head_layer, is_train=False)
        scaler_head_layer.load("static_inputs")


        ds_params = scaler_head_layer.transform_inverse(ds_params, "static_inputs", subset= list(ds_params.data_vars))


        ds_params.to_netcdf("/mnt/CEPH_PROJECTS/InterTwin/Wflow/models/emo1/run_default/inference_parameters.nc")

        return 