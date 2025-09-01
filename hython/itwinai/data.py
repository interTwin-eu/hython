# == common ==
# hython_trainer: str,
# dataset: str,
# data_lazy_load: bool,
# scaler: Dict,
# scaling_use_cached: bool,
# experiment_name: str,
# experiment_run: str,
# data_source: dict,
# work_dir: str,
# preprocessor: Dict,
# dynamic_inputs: List[str] | None = None,
# static_inputs: List[str] | None = None,
# target_variables: List[str] | None = None,
# calibration_target_variables: List[str] | None = None,
# head_output_variables: List[str] | None = None,
# scaling_static_range: Dict | None = None,
# scaling_rescale_target: Dict | None = None,
# mask_variables: List[str] | None = None,
# static_inputs_mask: List[str] | None = None,
# head_model_inputs: List[str] | None = None,
# train_temporal_range: List[str] = None,
# valid_temporal_range: List[str] = None,
# test_temporal_range: List[str] = None,
# train_downsampler: Dict | None = None,
# valid_downsampler: Dict | None = None,
# test_downsampler: Dict | None = None,
# downsampling_temporal_dynamic: bool | None = None,
# min_sample_target: int | None = None,
# target_has_missing_dates: bool | None = None,
# seq_length: int | None = None

from typing import Dict, List, Tuple
import xarray as xr
from itwinai.components import DataSplitter, monitor_exec

from hython.scaler import Scaler 
from hython.datasets import get_dataset
from hython.datasets.wflow_sbm import WflowSBM
from hython.config import Config


class RNNDatasetGetterAndPreprocessor(DataSplitter):
    def __init__(
        self,
        **kwargs
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))

    @monitor_exec
    def execute(self) -> Tuple[WflowSBM, WflowSBM, WflowSBM | None]:
        cfg = Config()

        
        configs = self.parameters["kwargs"]
        for i in configs:
            setattr(cfg, i, configs[i])

        scaler = Scaler(cfg, cfg.scaling_use_cached)
        
        # if "cal" in cfg.hython_trainer:
        #     cfg.target_variables = cfg.calibration_target_variables

        train_dataset = get_dataset(cfg.dataset)(cfg, scaler, True, "train")

        val_dataset = get_dataset(cfg.dataset)(cfg, scaler, False, "valid")

        test_dataset = get_dataset(cfg.dataset)(cfg, scaler, False, "test")
        

        return train_dataset, val_dataset, test_dataset
