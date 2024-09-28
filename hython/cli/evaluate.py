"""Description

"""

from jsonargparse import CLI  # type: ignore

from torch.utils.data import DataLoader  # type: ignore
import torch.optim as optim  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau  # type: ignore
from torch import nn  # type: ignore

from hython.models import *
from hython.datasets.datasets import get_dataset
from hython.sampler import *
from hython.normalizer import Normalizer
from hython.trainer import *
from hython.utils import (
    read_from_zarr,
    missing_location_idx,
    set_seed,
    prepare_for_plotting,
)
from hython.evaluator import predict
from hython.trainer import train_val
from hython.viz import *

import matplotlib.pyplot as plt


def evalute(
    # wflow model folder containing forcings, static parameters and outputs
    surr_input: str,
    surr_model: str,
    experiment: str,  # suffix of model weights file
    # paths to inputs and outputs
    dir_wflow_model: str,
    dir_wflow_input: str,
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
    metrics: dict,
):
    file_surr_input = f"{dir_surr_input}/{surr_input}"

    file_surr_model = f"{dir_surr_model}/{experiment}_{surr_model}"

    file_wflow_target = f"{dir_wflow_input}/{dir_wflow_model}/{file_target}"

    device = torch.device(device)

    train_temporal_range = slice(*train_temporal_range)
    valid_temporal_range = slice(*valid_temporal_range)

    # === READ TRAIN =============================================================

    Xs = read_from_zarr(url=file_surr_input, group="xs", multi_index="gridcell").xs.sel(
        feat=static_names
    )

    # === READ VALID. =============================================================

    Xd_test = (
        read_from_zarr(url=file_surr_input, group="xd", multi_index="gridcell")
        .sel(time=valid_temporal_range)
        .xd.sel(feat=dynamic_names)
    )
    Y_test = (
        read_from_zarr(url=file_surr_input, group="y", multi_index="gridcell")
        .sel(time=valid_temporal_range)
        .y.sel(feat=target_names)
    )

    # MASK
    masks = (
        read_from_zarr(url=file_surr_input, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    # === NORMALIZE  ========================================================================

    # TODO: avoid input normalization, compute stats and implement normalization of mini-batches

    normalizer_dynamic.read_stats(f"{dir_stats_output}/{experiment}_xd.npy")
    normalizer_static.read_stats(f"{dir_stats_output}/{experiment}_xs.npy")
    normalizer_target.read_stats(f"{dir_stats_output}/{experiment}_y.npy")

    # TODO: save stats, implement caching of stats to save computation

    Xd_test = normalizer_dynamic.normalize(Xd_test)
    Xs = normalizer_static.normalize(Xs)
    Y_test = normalizer_target.normalize(Y_test)

    # ==== MODEL ============================================================================

    model.to(device)

    # model load precomputed weights
    print(f"loading model {file_surr_model}")
    model.load_state_dict(torch.load(file_surr_model))

    # === PREDICT =============================================================================

    ds_target = (
        xr.open_dataset(file_wflow_target, chunks={"time": 200})
        .isel(lat=slice(None, None, -1))
        .sel(layer=1, drop=True)
    )

    lat, lon, time = len(masks.lat), len(masks.lon), Xd_test.shape[1]

    y_pred = predict(Xd_test.values, Xs.values, model, batch, device)

    y_pred = normalizer_target.denormalize(y_pred)

    Y_test = normalizer_target.denormalize(Y_test)
    # === EVALUATE ===============================================================================

    for iv, var in enumerate(target_names):
        metrics_var = metrics.pop(var)

        y_target_plot, y_pred_plot = prepare_for_plotting(
            y_target=Y_test[:, :, [iv]].values,
            y_pred=y_pred[:, :, [iv]],
            shape=(lat, lon, time),
            coords=ds_target.sel(time=valid_temporal_range).coords,
        )

        y_target_plot = y_target_plot.where(~masks.values[..., None])
        y_pred_plot = y_pred_plot.where(~masks.values[..., None])

        for metric in metrics_var:
            print(metric)
            if "rmse" in metric:
                fig, ax, rmse = map_rmse(
                    y_target_plot,
                    y_pred_plot,
                    unit="ET (mm)",
                    figsize=(8, 8),
                    return_rmse=True,
                )
            elif "kge" in metric:
                fig, ax, kge = map_kge(
                    y_target_plot,
                    y_pred_plot,
                    figsize=(12, 12),
                    return_kge=True,
                    kwargs_imshow={"vmin": -0.5, "vmax": 1},
                    ticks=np.linspace(-0.5, 1, 16),
                )
            elif "pbias" in metric:
                fig, ax, pbias = map_pbias(
                    y_target_plot,
                    y_pred_plot,
                    figsize=(12, 12),
                    return_pbias=True,
                    kwargs_imshow={"vmin": -100, "vmax": 100},
                    ticks=[l * 10 for l in range(-10, 11, 1)],
                )
            else:
                print(f"{metric} not found")
                continue

            fig.savefig(
                f"{dir_eval_output}/{experiment}_{surr_model.split('.')[0]}_{var}_{metric}.png"
            )

        ts_compare(
            y_target_plot,
            y_pred_plot,
            lat=[46.4],
            lon=[11.4],
            save=f"{dir_eval_output}/{experiment}_{surr_model.split('.')[0]}_{var}_ts_compare.png",
        )


if __name__ == "__main__":
    CLI(as_positional=False)
