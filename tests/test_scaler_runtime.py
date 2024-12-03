"""Test trainer"""
import pytest
import torch
import os


from hython.datasets import get_dataset
from hython.trainer import train_val, RNNTrainer
from hython.sampler import SamplerBuilder
from hython.utils import set_seed
from hython.models.cudnnLSTM import CuDNNLSTM
from hython.scaler import Scaler

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate


@pytest.mark.parametrize(
    "scale_at_runtime",
    [False, True],
)
def test_train(scale_at_runtime):
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/lstm_training.yaml"))

    set_seed(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_out_path = f"{cfg.run_dir}/{cfg.experiment_name}_{cfg.experiment_run}.pt"

    scaler = Scaler(cfg)

    train_dataset = get_dataset(cfg.dataset)(cfg, scaler, True, "train", scale_at_runtime)

    val_dataset = get_dataset(cfg.dataset)(cfg, scaler, False, "valid", scale_at_runtime)

    train_sampler_builder = SamplerBuilder(
        train_dataset, sampling="random", processing="single-gpu"
    )

    val_sampler_builder = SamplerBuilder(
        val_dataset, sampling="sequential", processing="single-gpu"
    )

    train_sampler = train_sampler_builder.get_sampler()
    val_sampler = val_sampler_builder.get_sampler()

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch, sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch, sampler=val_sampler)

    model = CuDNNLSTM(
        hidden_size=cfg.hidden_size,
        dynamic_input_size=len(cfg.dynamic_inputs),
        static_input_size=len(cfg.static_inputs),
        output_size=len(cfg.target_variables),
        dropout=cfg.dropout,
    )

    model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    trainer = RNNTrainer(
        cfg
    )

    model, loss_history, metric_history = train_val(
        trainer,
        model,
        train_loader,
        val_loader,
        cfg.epochs,
        opt,
        lr_scheduler,
        model_out_path,
        device,
    )


