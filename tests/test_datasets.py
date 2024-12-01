import pytest
import torch
import numpy as np
import xarray as xr
from pathlib import Path
import os
import random

import dask
from torch import nn

from hython.datasets import LSTMDataset, get_dataset
from hython.trainer import train_val, RNNTrainer, RNNTrainParams
from hython.sampler import SamplerBuilder, RegularIntervalDownsampler
from hython.metrics import MSEMetric
from hython.losses import RMSELoss
from hython.io import read_from_zarr
from hython.utils import set_seed
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.normalizer import Normalizer

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate


# configs
def test_train():
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/lstm_training.yaml"))
