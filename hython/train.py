from jsonargparse import CLI

from torch import nn
from hython.models import *
from hython.datasets.datasets import get_dataset
from hython.sampler import *
from hython.normalizer import Normalizer
from hython.trainer import *
from hython.utils import read_from_zarr, missing_location_idx, set_seed
from hython.train_val import train_val
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(
    # wflow model folder containing forcings, static parameters and outputs
    wflow_model: str,
    # paths to inputs and outputs
    dir_surr_input: str,
    dir_surr_output: str,
    dir_stats_output: str,
    dir_temporary_stuff: str,
    # variables name
    static_names: list,
    dynamic_names: list,
    target_names: list,
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
    #
    trainer: AbstractTrainer,  # TODO: Metric function should be a class
    #
    normalizer_static: Normalizer,
    normalizer_dynamic: Normalizer,
    normalizer_target: Normalizer,
):
    set_seed(seed)

    file_surr_input = f"{dir_surr_input}/{wflow_model}.zarr"

    device = torch.device(device)

    train_temporal_range = slice(*train_temporal_range)
    test_temporal_range = slice(*test_temporal_range)

    print(train_temporal_range)
    # SPLIT TRAIN
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
    TIME_RANGE = Xd.shape[1]

    # SPLIT TEST
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

    # MASK
    masks = (
        read_from_zarr(url=file_surr_input, group="mask")
        .mask.sel(mask_layer=mask_names)
        .any(dim="mask_layer")
    )

    print(masks)

    # NORMALIZE

    # TODO: avoid input normalization, compute stats and implement normalization of mini-batches

    normalizer_dynamic.compute_stats(Xd)
    normalizer_static.compute_stats(Xs)
    normalizer_target.compute_stats(Y)

    # TODO: save stats, implement caching of stats to save computation

    Xd = normalizer_dynamic.normalize(Xd)
    Xs = normalizer_static.normalize(Xs)
    Y = normalizer_target.normalize(Y)

    Xd_test = normalizer_dynamic.normalize(Xd_test)
    Y_test = normalizer_target.normalize(Y_test)

    # DATASET TODO: better way to convert xarray to torch tensor
    # LOOK: https://github.com/xarray-contrib/xbatcher
    train_dataset = get_dataset(dataset)(
        torch.Tensor(Xd.values), torch.Tensor(Y.values), torch.Tensor(Xs.values)
    )
    test_dataset = get_dataset(dataset)(
        torch.Tensor(Xd_test.values),
        torch.Tensor(Y_test.values),
        torch.Tensor(Xs.values),
    )

    # SAMPLER

    train_sampler_builder.initialize(
        shape=SHAPE, mask_missing=masks.values, torch_dataset=train_dataset
    )  # TODO: RandomSampler requires dataset torch
    test_sampler_builder.initialize(
        shape=SHAPE, mask_missing=masks.values, torch_dataset=test_dataset
    )

    train_sampler = train_sampler_builder.get_sampler()
    test_sampler = test_sampler_builder.get_sampler()

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch, sampler=test_sampler)

    # model
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    # build output name

    file_surr_output = f"{dir_surr_output}/test.pt"

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
        TIME_RANGE,
    )

    # save or plot loss and metric history
    # trainer.load_best_and_save_weights(model)


if __name__ == "__main__":
    CLI(as_positional=False)
