"""Description

"""

from jsonargparse import CLI # type: ignore

from torch.utils.data import DataLoader # type: ignore
import torch.optim as optim # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau # type: ignore
from torch import nn # type: ignore

from hython.models import *
from hython.datasets.datasets import get_dataset
from hython.sampler import *
from hython.normalizer import Normalizer
from hython.trainer import *
from hython.utils import read_from_zarr, missing_location_idx, set_seed, prepare_for_plotting
from hython.evaluator import predict
from hython.trainer import train_val
from hython.viz import map_rmse

import matplotlib.pyplot as plt


def evalute(
    # wflow model folder containing forcings, static parameters and outputs
    surr_input: str,
    surr_model: str,
    experiment: str, # suffix of model weights file
    # paths to inputs and outputs
    dir_wflow_model: str,
    dir_wflow_input:str,
    file_target: str,
    dir_surr_input: str,
    dir_surr_model: str,
    dir_stats_output: str,
    dir_eval_output: str,
    # variables name
    static_names: list,
    dynamic_names: list,
    target_names: list,
    mask_names: list,  # the layers that should mask out values from the training
    # train and test periods
    train_temporal_range: list,
    valid_temporal_range: list,
    # training parameters
    batch: int,
    device: str,

    model: nn.Module,
    #
    normalizer_static: Normalizer,
    normalizer_dynamic: Normalizer,
    normalizer_target: Normalizer,
    metrics: dict
):


    
    file_surr_input = f"{dir_surr_input}/{surr_input}"

    file_surr_model = f"{dir_surr_model}/{experiment}_{surr_model}"

    file_wflow_target = f"{dir_wflow_input}/{dir_wflow_model}/{file_target}"

    device = torch.device(device)

    train_temporal_range = slice(*train_temporal_range)
    valid_temporal_range = slice(*valid_temporal_range)

    # === READ TRAIN ============================================================= 
    Xd = (
        read_from_zarr(url=file_surr_input, group="xd", multi_index="gridcell")
        .sel(time=train_temporal_range)
        .xd.sel(feat=dynamic_names)
    )
    Xs = read_from_zarr(url=file_surr_input, group="xs", multi_index="gridcell").xs.sel(
        feat=static_names
    )
    Y = (
        read_from_zarr(url=file_surr_input, group="y", multi_index="gridcell")
        .sel(time=train_temporal_range)
        .y.sel(feat=target_names)
    )

    # === READ VALID. ============================================================= 

    Xd_valid = (
        read_from_zarr(url=file_surr_input, group="xd", multi_index="gridcell")
        .sel(time=valid_temporal_range)
        .xd.sel(feat=dynamic_names)
    )
    Y_valid = (
        read_from_zarr(url=file_surr_input, group="y", multi_index="gridcell")
        .sel(time=valid_temporal_range)
        .y.sel(feat=target_names)
    )

    SHAPE = Xd.attrs["shape"]
    TIME_RANGE = Xd.shape[1]

    # MASK
    masks = (
        read_from_zarr(url=file_surr_input, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    # === NORMALIZE  ========================================================================

    # TODO: avoid input normalization, compute stats and implement normalization of mini-batches

    normalizer_dynamic.compute_stats(Xd)
    normalizer_static.compute_stats(Xs)
    normalizer_target.compute_stats(Y)

    # TODO: save stats, implement caching of stats to save computation

    Xd_valid = normalizer_dynamic.normalize(Xd_valid)
    Xs = normalizer_static.normalize(Xs)
    Y_valid = normalizer_target.normalize(Y_valid)

    
    # ==== MODEL ============================================================================
    
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    # model load precomputed weights 
    model.load_state_dict(torch.load(file_surr_model))

    # === PREDICT =============================================================================
    
    ds_target = xr.open_dataset(file_wflow_target, chunks= {"time":200}).isel(lat=slice(None, None, -1)).sel(layer=1, drop=True)
    
    lat, lon, time = len(masks.lat),len(masks.lon), Xd_valid.shape[1]

    y_pred = predict(Xd_valid.values, Xs.values, model, batch, device)

    y_pred = normalizer_target.denormalize(y_pred)

    y_target_plot, y_pred_plot = prepare_for_plotting(y_target=Y_valid[:,:,[1]].values,
                                                  y_pred = y_pred[:,:,[1]], 
                                                  shape = (lat, lon, time), 
                                                  coords  = ds_target.sel(time=valid_temporal_range).coords)


    y_target_plot= y_target_plot.where(~masks.values[...,None])
    y_pred_plot = y_pred_plot.where(~masks.values[...,None])
    
    # === EVALUATE ===============================================================================

    eval_var = target_names[0]
    metric = metrics.pop(eval_var)[0]
    
    fig, ax, rmse = map_rmse(y_target_plot, y_pred_plot, unit = "ET (mm)", figsize = (8, 8), return_rmse=True)

    fig.savefig(f"{dir_eval_output}/{experiment}_{surr_model.split('.')[0]}_{metric}.png")


if __name__ == "__main__":
    CLI(as_positional=False)