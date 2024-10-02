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
    generate_model_name,
)
from hython.trainer import train_val


def train(
    # preprocessed inputs
    surr_input: str,
    dir_surr_input: str,
    # output name experiment + surr_model_output
    surr_model_output: str,
    experiment: str,
    # output directories
    dir_surr_output: str,
    dir_stats_output: str,
    # variables name
    static_names: list,
    dynamic_names: list,
    target_names: list,
    # masks
    mask_names: list,  # the layers that should mask out values from the training
    # train and test periods
    train_temporal_range: list,
    test_temporal_range: list,
    # training parameters
    epochs: int,
    batch: int,
    seed: int,
    device: str,
    # train and test samplers
    train_sampler_builder: SamplerBuilder,
    test_sampler_builder: SamplerBuilder,
    # torch dataset
    dataset: str,
    # NN model
    model: nn.Module,
    # Trainer
    trainer: AbstractTrainer,  # TODO: Metric function should be a class
    # Normalizer
    normalizer_static: Normalizer,
    normalizer_dynamic: Normalizer,
    normalizer_target: Normalizer,
):
    set_seed(seed)

    file_surr_input = f"{dir_surr_input}/{surr_input}"

    device = torch.device(device)

    train_temporal_range = slice(*train_temporal_range)
    test_temporal_range = slice(*test_temporal_range)

    # === READ TRAIN ===========================================================

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

    SHAPE = Xd.attrs["shape"]

    # === READ TEST ===================================================================

    Y_test = (
        read_from_zarr(url=file_surr_input, group="y", multi_index="gridcell")
        .sel(time=test_temporal_range)
        .y.sel(feat=target_names)
    )
    Xd_test = (
        read_from_zarr(url=file_surr_input, group="xd", multi_index="gridcell")
        .sel(time=test_temporal_range)
        .xd.sel(feat=dynamic_names)
    )

    # === MASK ============================================================================
    masks = (
        read_from_zarr(url=file_surr_input, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    # === SAMPLER ========================================================================

    train_sampler_builder.initialize(
        shape=SHAPE, mask_missing=masks.values
    )  # TODO: RandomSampler requires dataset torch
    test_sampler_builder.initialize(shape=SHAPE, mask_missing=masks.values)

    train_sampler = train_sampler_builder.get_sampler()
    test_sampler = test_sampler_builder.get_sampler()

    # === NORMALIZE ============================================================================

    # TODO: avoid input normalization, compute stats and implement normalization of mini-batches

    # TODO: save stats, implement caching of stats to save computation

    normalizer_dynamic.compute_stats(Xd[train_sampler_builder.indices])
    normalizer_static.compute_stats(Xs[train_sampler_builder.indices])
    normalizer_target.compute_stats(Y[train_sampler_builder.indices])

    Xd = normalizer_dynamic.normalize(
        Xd, write_to=f"{dir_stats_output}/{experiment}_xd.npy"
    )
    Xs = normalizer_static.normalize(
        Xs, write_to=f"{dir_stats_output}/{experiment}_xs.npy"
    )
    Y = normalizer_target.normalize(
        Y, write_to=f"{dir_stats_output}/{experiment}_y.npy"
    )

    Xd_test = normalizer_dynamic.normalize(Xd_test)
    Y_test = normalizer_target.normalize(Y_test)

    # === DATASET ========================================================================

    # TODO: find better way to convert xarray to torch tensor
    # LOOK: https://github.com/xarray-contrib/xbatcher

    train_dataset = get_dataset(dataset)(
        torch.Tensor(Xd.values), torch.Tensor(Y.values), torch.Tensor(Xs.values)
    )

    test_dataset = get_dataset(dataset)(
        torch.Tensor(Xd_test.values),
        torch.Tensor(Y_test.values),
        torch.Tensor(Xs.values),
    )

    # === DATALOADER =====================================================================

    train_loader = DataLoader(train_dataset, batch_size=batch, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch, sampler=test_sampler)

    # === MODEL =========================================================================
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    # === TRAIN ===========================================================================
    # TODO: build output name

    file_surr_output = f"{dir_surr_output}/{experiment}_{surr_model_output}"

    # train
    model, loss_history, metric_history = train_val(
        trainer,
        model,
        train_loader,
        test_loader,
        epochs,
        opt,
        lr_scheduler,
        file_surr_output,
        device,
    )

    # === METRICS ====================================================================


if __name__ == "__main__":
    CLI(as_positional=False)
