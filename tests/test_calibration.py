"""Test cal"""
import pytest
import torch

import os

from hython.datasets import get_dataset
from hython.trainer import train_val,CalTrainer 
from hython.sampler import SamplerBuilder

from hython.utils import set_seed
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.models.transferNN import TransferNN
from hython.models.hybrid import Hybrid
from hython.scaler import Scaler

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

# configs 
def test_cal():
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/calibration.yaml"))

    set_seed(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_out_path = f"{cfg.data_dir}/{cfg.experiment_name}_{cfg.experiment_run}.pt"

    scaler = Scaler(cfg)

    train_dataset = get_dataset(cfg.dataset)(
            cfg, scaler, True, "train"
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

    surrogate = CuDNNLSTM(
                    hidden_size=cfg.model_head_hidden_size, 
                    dynamic_input_size=len(cfg.dynamic_inputs),
                    static_input_size=len(cfg.head_model_inputs), 
                    output_size=len(cfg.target_variables),
                    dropout=cfg.model_head_dropout
    )

    surrogate.load_state_dict(torch.load(f"{cfg.model_head_dir}/{cfg.model_head_file}"))

    transfer_nn = TransferNN( len(cfg.static_inputs), len(cfg.head_model_inputs) ).to(device)
    
    model = Hybrid( 
                transfernn=transfer_nn,
                head_layer=surrogate,
                freeze_head=cfg.freeze_head,
                scale_head_input_parameter=cfg.scale_head_input_parameter
    ).to(device)


    # opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    
    trainer = CalTrainer(cfg)

    model, loss_history, metric_history = train_val(
        trainer,
        model,
        train_loader,
        val_loader,
        cfg.epochs,
        #opt,
        #lr_scheduler,
        model_out_path,
        device
    )
