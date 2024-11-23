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
from hython.normalizer import Normalizer

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


EXP_METADATA = {
    "vwc_2":[ "thetaS", "thetaR"],
    "actevap_2":["Swood", "Sl"] ,
    "vwc_4":[ "thetaS", "thetaR", "KsatVer", "SoilThickness"],
    "vwc_4_1":[ "thetaS", "thetaR", "KsatVer", "SoilThickness"], # train more data and longer epochs
    "vwc_4_2":[ "thetaS", "thetaR", "KsatVer", "SoilThickness"], # train with deeper layers,
    "vwc_4_3":[ "thetaS", "thetaR", "KsatVer", "SoilThickness"], # train with deeper smaller kernel 3 x 3
    "vwc_4_4":[ "thetaS", "thetaR", "KsatVer", "SoilThickness"], # simple LSTM
    "vwc_actevap":['thetaS', 'thetaR', 'SoilThickness','RootingDepth', 'Swood','KsatVer', 'Sl', 'f', 'Kext', 'PathFrac', 'WaterFrac'],
    "vwc_cal": {"static":['thetaS', 'thetaR', "KsatVer", 'SoilThickness','RootingDepth'],
                "target":[ "vwc"] },
    "exp_landflux":{"static":['thetaS', 'thetaR', "KsatVer", 'SoilThickness','RootingDepth'], # 'Slope','Swood',' Sl', 'f', 'Kext' 
                    "target":[ "recharge", "runoff_land", "runoff_river"]} #, "ssf", "vwc", "actevap" #"runoff_land", "runoff_river",
}


EXPERIMENT  = "vwc_cal"

SURROGATE_INPUT = "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/adg1km_eobs_preprocessed.zarr/" #"https://eurac-eo.s3.amazonaws.com/INTERTWIN/SURROGATE_INPUT/adg1km_eobs_preprocessed.zarr/"

SURROGATE_MODEL_OUTPUT = f"/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_model/{EXPERIMENT}.pt"
TMP_STATS = "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_stats"

# === FILTER ==============================================================

# train/test temporal range
train_temporal_range = slice("2015-01-01","2018-12-31")
test_temporal_range = slice("2019-01-01", "2019-12-31")

# variables
dynamic_names = ["precip", "pet", "temp"] 
static_names = EXP_METADATA[EXPERIMENT]["static"]
target_names = EXP_METADATA[EXPERIMENT]["target"]  #"q_land_toriver", "ssf_toriver","runoff_river"] #, "runoff_land", "runoff_river", "snow", "snowwater"]#, "snow", "snowwater"]

# === MASK ========================================================================================

mask_names = ["mask_missing", "mask_lake"] # names depends on preprocessing application

# === DATASET ========================================================================================

DATASET = "LSTMDataset"

# == MODEL  ========================================================================================

HIDDEN_SIZE = 36
DYNAMIC_INPUT_SIZE = len(dynamic_names)
STATIC_INPUT_SIZE = len(static_names)
OUTPUT_SIZE = len(target_names)

TARGET_WEIGHTS = {t:1/len(target_names) for t in target_names}


# === SAMPLER/TRAINER ===================================================================================

EPOCHS = 3
BATCH = 256
SEED = 42

# downsampling
DOWNSAMPLING = True
TRAIN_INTERVAL = [3,3]
TRAIN_ORIGIN = [0,0]

TEST_INTERVAL = [3,3]
TEST_ORIGIN = [2,2]

TEMPORAL_SUBSAMPLING = True
TEMPORAL_SUBSET = [200, 150] 
SEQ_LENGTH = 120


assert sum(v for v in TARGET_WEIGHTS.values()) == 1, "check target weights"
TARGET_INITIALS = "".join([i[0].capitalize() for i in target_names])

set_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Xd = (
    read_from_zarr(url=SURROGATE_INPUT, group="xd", multi_index="gridcell")
    .sel(time=train_temporal_range)
    .xd.sel(feat=dynamic_names)
)
Xs = read_from_zarr(url=SURROGATE_INPUT, group="xs", multi_index="gridcell").xs.sel(
    feat=static_names
)
Y = (
    read_from_zarr(url=SURROGATE_INPUT, group="y", multi_index="gridcell")
    .sel(time=train_temporal_range)
    .y.sel(feat=target_names)
)

SHAPE = Xd.attrs["shape"]


# === READ TEST ===================================================================

Y_test = (
    read_from_zarr(url=SURROGATE_INPUT, group="y", multi_index="gridcell")
    .sel(time=test_temporal_range)
    .y.sel(feat=target_names)
)
Xd_test = (
    read_from_zarr(url=SURROGATE_INPUT, group="xd", multi_index="gridcell")
    .sel(time=test_temporal_range)
    .xd.sel(feat=dynamic_names)

)



masks = (
    read_from_zarr(url=SURROGATE_INPUT, group="mask")
    .mask.sel(mask_layer=mask_names)
    .any(dim="mask_layer")
)


if DOWNSAMPLING:
    train_downsampler = RegularIntervalDownsampler(
        intervals=TRAIN_INTERVAL, origin=TRAIN_ORIGIN
    )       
    test_downsampler = RegularIntervalDownsampler(
        intervals=TEST_INTERVAL, origin=TEST_ORIGIN
    )
else:
    train_downsampler, test_downsampler = None, None



method = "minmax"

normalizer_dynamic = Normalizer(method=method,
                                type="spacetime", 
                                axis_order="NTC",
                                save_stats=f"{TMP_STATS}/{EXPERIMENT}_xd.npy")
normalizer_static = Normalizer(method=method,
                               type="space", 
                               axis_order="NTC",
                               save_stats=f"{TMP_STATS}/{EXPERIMENT}_xs.npy")
normalizer_target = Normalizer(method=method, 
                               type="spacetime",
                               axis_order="NTC",
                               save_stats=f"{TMP_STATS}/{EXPERIMENT}_y.npy")

def main():

    train_dataset = get_dataset(DATASET)(
            Xd,
            Y,
            Xs,
            original_domain_shape=SHAPE,
            mask=masks,
            downsampler=train_downsampler,
            normalizer_dynamic=normalizer_dynamic,
            normalizer_static=normalizer_static,
            normalizer_target=normalizer_target,
            #persist=True
    )



    test_dataset = get_dataset(DATASET)(
            Xd_test,
            Y_test,
            Xs,
            original_domain_shape=SHAPE,
            mask=masks,
            downsampler=test_downsampler,
            normalizer_dynamic=normalizer_dynamic,
            normalizer_static=normalizer_static,
            normalizer_target=normalizer_target, 
            #persist=True
    )




    train_sampler_builder = SamplerBuilder(
        train_dataset,
        sampling="random", 
        processing="single-gpu")

    test_sampler_builder = SamplerBuilder(
        test_dataset,
        sampling="sequential", 
        processing="single-gpu")


    train_sampler = train_sampler_builder.get_sampler()
    test_sampler = test_sampler_builder.get_sampler()

    train_loader = DataLoader(train_dataset, batch_size=BATCH , sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH , sampler=test_sampler)



    model = CuDNNLSTM(
                    hidden_size=HIDDEN_SIZE, 
                    dynamic_input_size=DYNAMIC_INPUT_SIZE,
                    static_input_size=STATIC_INPUT_SIZE, 
                    output_size=OUTPUT_SIZE,
                    #dropout=0.2
    )

    model.to(device)


    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    #loss_fn = RMSELoss(target_weight=TARGET_WEIGHT)

    loss_fn = RMSELoss(target_weight=TARGET_WEIGHTS)

    metric_fn = MSEMetric()

    trainer = RNNTrainer(
        RNNTrainParams(
                experiment=EXPERIMENT,
                temporal_subsampling=TEMPORAL_SUBSAMPLING, 
                temporal_subset=TEMPORAL_SUBSET, 
                seq_length=SEQ_LENGTH, 
                target_names=target_names,
                metric_func=metric_fn,
                loss_func=loss_fn,
                #loss_physics_collection=physics_loss_collection
        ))

    model, loss_history, metric_history = train_val(
        trainer,
        model,
        train_loader,
        test_loader,
        EPOCHS,
        opt,
        lr_scheduler,
        SURROGATE_MODEL_OUTPUT,
        device
    )

if __name__ == "__main__":
    main()