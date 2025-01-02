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
        dataset: str,
        scaling_variant: str,
        experiment_name: str, 
        experiment_run: str,
        data_dir: str,
        data_file: str,
        run_dir: str,
        surrogate_input: str,
        dynamic_inputs: list[str],
        static_inputs: list[str],
        target_variables: list[str],
        mask_variables: list[str],
        train_temporal_range: list[str] = ["", ""],
        valid_temporal_range: list[str] = ["", ""],
        train_downsampler: dict = None,
        valid_downsampler: dict = None
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))

        self.cfg = deepcopy(self.locals2params(locals()))


        self.cfg = instantiate(OmegaConf.create(self.cfg))

    @monitor_exec
    def execute(self) -> Tuple[Wflow1d, Wflow1d, None]:

        scaler = Scaler(self.cfg)

        train_dataset = get_dataset(self.cfg.dataset)(self.cfg, scaler, True, "train")

        val_dataset = get_dataset(self.cfg.dataset)(self.cfg, scaler, False, "valid")

        return train_dataset, val_dataset, None

