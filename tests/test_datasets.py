import pytest
import torch
import numpy as np
import xarray as xr
from pathlib import Path
import os

from hython.sampler import SamplerBuilder
from hython.datasets import Wflow2d, Wflow2dCal
from hython.trainer import ConvTrainer
from hython.io import read_from_zarr
from hython.models.convLSTM import ConvLSTM
from hython.scaler import Scaler

import os
from omegaconf import OmegaConf
from hydra.utils import instantiate



def test_wflow2d():
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/convlstm_training.yaml"))

    scaler = Scaler(cfg)

    dataset = Wflow2d(cfg, scaler, True, "train")

    train_sampler_builder = SamplerBuilder(
        dataset, sampling="random", processing="single-gpu"
    )

    train_sampler = train_sampler_builder.get_sampler()

    for i in train_sampler:
        i

def test_wflow2dcal():
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/convlstm_calibration.yaml"))

    scaler = Scaler(cfg)

    dataset = Wflow2dCal(cfg, scaler, True, "train")

    train_sampler_builder = SamplerBuilder(
        dataset, sampling="random", processing="single-gpu"
    )

    train_sampler = train_sampler_builder.get_sampler()


    for i in train_sampler:
        i
    