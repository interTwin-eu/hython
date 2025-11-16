import pytest
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hython.datasets import get_dataset
from hython.scaler import Scaler
from torch.utils.data import DataLoader
from hython.models import HBV
import os

def test_pbm():
    cfg = instantiate(OmegaConf.load(f"{os.path.dirname(__file__)}/config/pbm.yaml"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = Scaler(cfg)


    train_dataset = get_dataset(cfg.dataset)(cfg, scaler, True, "train")

    train_loader = DataLoader(  
        train_dataset, batch_size=cfg.batch
    )
    hbv = HBV()

    for data in train_loader:
        x_conceptual= data["xd"]
        hbv(x_conceptual)

