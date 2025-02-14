from typing import Optional, Tuple, Any
import xarray as xr
from itwinai.components import DataProcessor, DataSplitter, monitor_exec
from hython.io import read_from_zarr
from hython.datasets import get_dataset
from hython.datasets.wflow_sbm import Wflow1d
from hython.scaler import Scaler
from hython.sampler.downsampler import AbstractDownSampler

from omegaconf import OmegaConf
from hydra.utils import instantiate

from copy import deepcopy


class RNNDatasetGetterAndPreprocessor(DataSplitter):
    def __init__(
        self,
        # == common ==
        hython_trainer: str,
        dataset: str,
        downsampling_temporal_dynamic: bool,
        data_lazy_load: bool,
        scaling_variant: str,
        scaling_use_cached: bool,
        experiment_name: str,
        experiment_run: str,
        data_source: dict,
        work_dir: str,
        dynamic_inputs: list[str] = None,
        static_inputs: list[str] = None,
        target_variables: list[str] = None,
        scaling_static_range: dict = None,
        mask_variables: list[str] = None,
        static_inputs_mask: list[str] = None,
        head_model_inputs: list[str] = None,
        train_temporal_range: list[str] = ["", ""],
        valid_temporal_range: list[str] = ["", ""],
        cal_temporal_range: list[str] = ["", ""],
        train_downsampler: dict = None,
        valid_downsampler: dict = None,
        cal_downsampler: dict = None,
        # == calibration ==
        # data_dynamic_inputs: str = None,
        # data_static_inputs: str = None,
        # data_target_variables: str = None,
        # data_target_mask: str = None,
        min_sample_target: int = None,
        seq_length: int = None,
        # == training ==
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))

        self.cfg = deepcopy(self.locals2params(locals()))

        self.cfg = instantiate(OmegaConf.create(self.cfg))

    @monitor_exec
    def execute(self) -> Tuple[Wflow1d, Wflow1d, None]:

        scaler = Scaler(self.cfg)
        
        period = "train"
        istrain = True
        if "cal" in self.cfg.hython_trainer:
            period = "cal" 
        
        train_dataset = get_dataset(self.cfg.dataset)(self.cfg, scaler, istrain, period)

        val_dataset = get_dataset(self.cfg.dataset)(self.cfg, scaler, False, "valid")
        
        return train_dataset, val_dataset, None
