import os
from pathlib import Path
from timeit import default_timer
from typing import Dict, Literal, Optional, Union, Any, Tuple
from tqdm.auto import tqdm
import copy
from torch.utils.data import Dataset, Dataloader
import torch
import torch.nn as nn
import pandas as pd
from ray import train


from hython.sampler import SamplerBuilder
from hython.trainer import RNNTrainer, CalTrainer
from hython.models import get_model_class as get_hython_model
from hython.models import load_model, ModelLogAPI
from itwinai.components import monitor_exec


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

from omegaconf import OmegaConf
from hydra.utils import instantiate



class ParameterInference(TorchPredictor):

    def __init__(self, 
                 test_dataset: Dataset, 
                 test_dataloader: Dataloader, 
                 model: Union[nn.Module, ModelLoader]):
        super.__init__(self, 
                       model = model, 
                       test_dataset=test_dataset, 
                       test_dataloader=test_dataloader )
        self.save_parameters(**self.locals2params(locals()))
        self.model = self.model.eval()

    @monitor_exec
    def execute(
        self,
        test_dataset: Dataset,
        model: nn.Module = None,
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
    
        if model is not None:
            # Overrides existing "internal" model
            self.model = model

        test_dataloader = self.test_dataloader_class(
            test_dataset, **self.test_dataloader_kwargs
        )

        all_predictions = dict()
        for samples_ids, samples in test_dataloader:
            with torch.no_grad():
                pred = self.model(samples)
            pred = self.transform_predictions(pred)
            for idx, pre in zip(samples_ids, pred):
                # For each item in the batch
                if pre.numel() == 1:
                    pre = pre.item()
                else:
                    pre = pre.to_dense().tolist()
                all_predictions[idx] = pre
        return all_predictions

class TrainingTest(TorchPredictor):

    def __init__(self, 
                 test_dataset: Dataset, 
                 test_dataloader: Dataloader, 
                 model: Union[nn.Module, ModelLoader]):
        super.__init__(self, 
                       model = model, 
                       test_dataset=test_dataset, 
                       test_dataloader=test_dataloader )
        self.save_parameters(**self.locals2params(locals()))
        self.model = self.model.eval()

    @monitor_exec
    def execute(
        self,
        test_dataset: Dataset,
        model: nn.Module = None,
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
    
        if model is not None:
            # Overrides existing "internal" model
            self.model = model

        test_dataloader = self.test_dataloader_class(
            test_dataset, **self.test_dataloader_kwargs
        )

        all_predictions = dict()
        for samples_ids, samples in test_dataloader:
            with torch.no_grad():
                pred = self.model(samples)
            pred = self.transform_predictions(pred)
            for idx, pre in zip(samples_ids, pred):
                # For each item in the batch
                if pre.numel() == 1:
                    pre = pre.item()
                else:
                    pre = pre.to_dense().tolist()
                all_predictions[idx] = pre
        return all_predictions
    
class CalibrationTest(TorchPredictor):

    def __init__(self, 
                 test_dataset: Dataset, 
                 test_dataloader: Dataloader, 
                 model: Union[nn.Module, ModelLoader]):
        super.__init__(self, 
                       model = model, 
                       test_dataset=test_dataset, 
                       test_dataloader=test_dataloader )
        self.save_parameters(**self.locals2params(locals()))
        self.model = self.model.eval()

    @monitor_exec
    def execute(
        self,
        test_dataset: Dataset,
        model: nn.Module = None,
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
    
        if model is not None:
            # Overrides existing "internal" model
            self.model = model

        test_dataloader = self.test_dataloader_class(
            test_dataset, **self.test_dataloader_kwargs
        )

        all_predictions = dict()
        for samples_ids, samples in test_dataloader:
            with torch.no_grad():
                pred = self.model(samples)
            pred = self.transform_predictions(pred)
            for idx, pre in zip(samples_ids, pred):
                # For each item in the batch
                if pre.numel() == 1:
                    pre = pre.item()
                else:
                    pre = pre.to_dense().tolist()
                all_predictions[idx] = pre
        return all_predictions
    
class SeasonalPrediction(TorchPredictor):

    def __init__(self, 
                 test_dataset: Dataset, 
                 test_dataloader: Dataloader, 
                 model: Union[nn.Module, ModelLoader]):
        super.__init__(self, 
                       model = model, 
                       test_dataset=test_dataset, 
                       test_dataloader=test_dataloader )
        self.save_parameters(**self.locals2params(locals()))
        self.model = self.model.eval()

    @monitor_exec
    def execute(
        self,
        test_dataset: Dataset,
        model: nn.Module = None,
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
    
        if model is not None:
            # Overrides existing "internal" model
            self.model = model

        test_dataloader = self.test_dataloader_class(
            test_dataset, **self.test_dataloader_kwargs
        )

        all_predictions = dict()
        for samples_ids, samples in test_dataloader:
            with torch.no_grad():
                pred = self.model(samples)
            pred = self.transform_predictions(pred)
            for idx, pre in zip(samples_ids, pred):
                # For each item in the batch
                if pre.numel() == 1:
                    pre = pre.item()
                else:
                    pre = pre.to_dense().tolist()
                all_predictions[idx] = pre
        return all_predictions