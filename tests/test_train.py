"""Test trainer"""
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
from hython.trainer import train_val,RNNTrainer, RNNTrainParams
from hython.sampler import SamplerBuilder, RegularIntervalDownsampler
from hython.metrics import MSEMetric
from hython.losses import RMSELoss
from hython.io import read_from_zarr
from hython.utils import set_seed
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.normalizer import Normalizer, Scaler

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

# configs 
def test_train():
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/lstm_training.yaml"))

    set_seed(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    file_path = f"{cfg.data_dir}/{cfg.data_file}"

    model_out_path = f"{cfg.data_dir}/{cfg.experiment_name}_{cfg.experiment_run}.pt"

    # Xd = (
    #     read_from_zarr(url=file_path, group="xd", multi_index="gridcell")
    #     .sel(time=slice(*cfg.train_temporal_range))
    #     .xd.sel(feat=cfg.dynamic_inputs)
    # )
    # Xs = read_from_zarr(url=file_path, group="xs", multi_index="gridcell").xs.sel(
    #     feat=cfg.static_inputs
    # )
    # Y = (
    #     read_from_zarr(url=file_path, group="y", multi_index="gridcell")
    #     .sel(time=slice(*cfg.train_temporal_range))
    #     .y.sel(feat=cfg.target_variables)
    # )

    # SHAPE = Xd.attrs["shape"]


    # # === READ TEST ===================================================================

    # Y_test = (
    #     read_from_zarr(url=file_path, group="y", multi_index="gridcell")
    #     .sel(time=slice(*cfg.valid_temporal_range))
    #     .y.sel(feat=cfg.target_variables)
    # )
    # Xd_test = (
    #     read_from_zarr(url=file_path, group="xd", multi_index="gridcell")
    #     .sel(time=slice(*cfg.valid_temporal_range))
    #     .xd.sel(feat=cfg.dynamic_inputs)

    # )

    # masks = (
    #     read_from_zarr(url=file_path, group="mask")
    #     .mask.sel(mask_layer=cfg.mask_variables)
    #     .any(dim="mask_layer")
    # )



    method = cfg.scaling_variant

    # normalizer_dynamic = Normalizer(method=method,
    #                                 type="spacetime", 
    #                                 axis_order="NTC")
    # normalizer_static = Normalizer(method=method,
    #                             type="space", 
    #                             axis_order="NTC")

    # normalizer_target = Normalizer(method=method, 
    #                             type="spacetime",
    #                             axis_order="NTC")

    scaler = Scaler(cfg)

    train_dataset = get_dataset(cfg.dataset)(
            cfg, scaler, True
    )


    val_dataset = get_dataset(cfg.dataset)(
            cfg, scaler, False, "valid"
    )


    train_sampler_builder = SamplerBuilder(
        train_dataset,
        sampling="random", 
        processing="single-gpu")

    val_sampler_builder = SamplerBuilder(
        val_dataset,
        sampling="sequential", 
        processing="single-gpu")


    train_sampler = train_sampler_builder.get_sampler()
    val_sampler = val_sampler_builder.get_sampler()

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch , sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch , sampler=val_sampler)


    model = CuDNNLSTM(
                    hidden_size=cfg.hidden_size, 
                    dynamic_input_size=len(cfg.dynamic_inputs),
                    static_input_size=len(cfg.static_inputs), 
                    output_size=len(cfg.target_variables),
                    dropout=cfg.dropout
    )

    model.to(device)


    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    
    trainer = RNNTrainer(
        RNNTrainParams(
                temporal_subsampling=cfg.temporal_downsampling, 
                temporal_subset=cfg.temporal_subset, 
                seq_length=cfg.seq_length, 
                target_names=cfg.target_variables,
                metric_func=cfg.metric_fn,
                loss_func=cfg.loss_fn,
        ))

    model, loss_history, metric_history = train_val(
        trainer,
        model,
        train_loader,
        val_loader,
        cfg.epochs,
        opt,
        lr_scheduler,
        model_out_path,
        device
    )
