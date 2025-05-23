import os, random
import numpy as np
import xarray as xr
import torch

from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

import datetime
from typing import Any
from numpy.typing import NDArray
from dask.array.core import Array as DaskArray
import itertools

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam


def generate_time_idx(time_idx, time_size, seq_len, cell_size):
    """Generate temporal indices to subset the temporal dimension of the spacetime_index.
       It is supposed to be used at runtime, to randomly subset the spacetime_index."""
    o = []
    for c in range(cell_size):
        for i in time_idx:
            index = i + c*time_size
            o.append(index)
    return o


def generate_run_folder(cfg):
    return f"{cfg.work_dir}/{generate_experiment_id(cfg)}/" # /{generate_timestamp()}


def generate_experiment_id(cfg):
    return "_".join([cfg.experiment_name, cfg.experiment_run])

def generate_timestamp():
    now = datetime.datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    second = f"{now.second}".zfill(2)
    timestamp = f'{day}{month}_{hour}{minute}{second}'
    return timestamp

def generate_model_name(surr_model_prefix, experiment, target_names, hidden_size, seed):
    TARGET_INITIALS = "".join([i[0].capitalize() for i in target_names])
    return (
        f"{surr_model_prefix}_{experiment}_v{TARGET_INITIALS}_h{hidden_size}_s{seed}.pt"
    )


def reclass(arr, classes):
    """Returns a 2D array with reclassified values

    Parameters
    ----------
    arr: NDArray | xr.DataArray
        The input array to be reclassified
    classes: List[int,float]

    Returns
    -------
    """
    if isinstance(arr, xr.DataArray):
        for ic in range(len(classes)):
            print(ic, len(classes) - 1)
            if ic < len(classes) - 1:
                arr = arr.where(~((arr >= classes[ic]) & (arr < classes[ic + 1])), ic)
            else:
                arr = arr.where(~(arr >= classes[ic]), ic)
    return arr


def load(surrogate_input_path, wflow_model, files=["Xd", "Xs", "Y"]):
    loaded = np.load(surrogate_input_path / f"{wflow_model}.npz")
    return [loaded[f] for f in files]


def missing_location_idx(
    grid: np.ndarray | xr.DataArray | xr.Dataset, missing: Any = np.nan
) -> NDArray | list:
    """Returns the indices corresponding to missing values

    Args:
        grid (np.ndarray | xr.DataArray | xr.Dataset): _description_
        missing (Any, optional): _description_. Defaults to np.nan.

    Returns:
        np.array | list: _description_
    """

    if isinstance(grid, np.ndarray) or isinstance(grid, torch.Tensor):
        location_idx = np.isnan(grid).any(axis=-1)

    elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
        pass
    else:
        pass

    return location_idx  # (gridcells, dims)


def build_mask_dataarray(masks: list, names: list = None):
    das = []
    for (
        mask,
        name,
    ) in zip(masks, names):
        das.append(mask.rename(name))
    return xr.merge(das).to_dataarray(dim="mask_layer", name="mask")


def to_xr(arr, coords, dims=["lat", "lon", "time"]):
    return xr.DataArray(arr, dims=dims, coords=coords)


def reconstruct_from_missing(
    a: NDArray, original_shape: tuple, missing_location_idx: NDArray
) -> NDArray:
    """Re-insert missing values where they were removed, based on the missing_location_idx

    Args:
        a (NDArray): The array without missing values.
        original_shape (tuple): The array shape before the missing values were removed.
        missing_location_idx (NDArray): The location (grid cell ids) of missing values

    Returns:
        NDArray: A new array filled with missing values
    """
    a_new = np.empty(original_shape)

    fill = np.full(
        (
            int(np.sum(missing_location_idx)),
            *(original_shape[1:] if len(original_shape) > 2 else [original_shape[1]]),
        ),
        np.nan,
    )

    if len(original_shape) > 2:
        # fill missing
        a_new[missing_location_idx, :, :] = fill

        # fill not missing
        a_new[~missing_location_idx, :, :] = a.copy()
    else:
        # fill missing
        a_new[missing_location_idx, :] = fill

        # fill not missing
        a_new[~missing_location_idx, :] = a.copy()

    return a_new


def reshape_to_2Dspatial2(a, lat_size, lon_size, feat_size, coords=None):
    tmp = a.reshape(lat_size, lon_size, feat_size)
    return tmp


def reshape_to_2Dspatial(a, lat_size, lon_size, time_size, feat_size, coords=None):
    tmp = a.reshape(lat_size, lon_size, time_size, feat_size)
    return tmp


def prepare_for_plotting2d(
    y_target,
    shape,
    coords,
):
    def to_xr(arr, coords, dims=["lat", "lon", "time"]):
        return xr.DataArray(arr, dims=dims, coords=coords)

    lat, lon = shape
    n_feat = y_target.shape[-1]

    y = reshape_to_2Dspatial2(y_target, lat, lon, n_feat)

    y = to_xr(y, coords=coords, dims=["lat", "lon", "variable"])

    return y


def prepare_for_plotting1d(
    y_target,
    shape,
    coords,
    reverse_lat = False
):
    lat, lon, time = shape
    n_feat = y_target.shape[-1]

    y = reshape_to_2Dspatial2(y_target, lat, lon, time, n_feat)

    print(y.shape)
    if reverse_lat:
        y = y[::-1]

    def to_xr(arr, coords, dims=["lat", "lon", "time"]):
        return xr.DataArray(arr, dims=dims, coords=coords)

    y = to_xr(y, coords=coords, dims=["lat", "lon", "time"])

    return y


def prepare_for_plotting(
    y_target: NDArray,
    y_pred: NDArray,
    shape: tuple[int],
    coords: DataArrayCoordinates | DatasetCoordinates,
    reverse_lat: bool = False
):
    lat, lon, time = shape
    n_feat = y_target.shape[-1]

    y = reshape_to_2Dspatial(y_target, lat, lon, time, n_feat)

    yhat = reshape_to_2Dspatial(y_pred, lat, lon, time, n_feat)

    y = to_xr(y[..., 0], coords=coords)
    yhat = to_xr(yhat[..., 0], coords=coords)

    return y, yhat


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_grid_indices(shape=None, grid=None):
    if grid is not None:
        if isinstance(grid, np.ndarray):
            shape = grid.shape
        elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
            shape = (len(grid.lat), len(grid.lon))
        else:
            pass

    ishape = shape[0]  # rows (y, lat)
    jshape = shape[1]  # columns (x, lon)

    grid_idx = np.arange(0, ishape * jshape, 1).reshape(ishape, jshape)

    return grid_idx


def compute_cubelet_spatial_idxs(
    shape,
    xsize,
    ysize,
    xover,
    yover,
    keep_degenerate_cbs=False,
    masks=None,
    missing_policy="all",
):  # assume time,lat,lon
    time_size, lat_size, lon_size = shape

    # compute
    space_idx = compute_grid_indices(shape=(lat_size, lon_size))

    idx = 0
    cbs_indexes, cbs_indexes_missing, cbs_indexes_degenerate, cbs_slices = (
        [],
        [],
        [],
        [],
    )

    for ix in range(0, lon_size, xsize - xover):
        for iy in range(0, lat_size, ysize - yover):
            xslice = slice(ix, ix + xsize)
            yslice = slice(iy, iy + ysize)
            # don't need the original data, but a derived 2D array of indices, very light!
            cubelet = space_idx[yslice, xslice]

            # decide whether keep or not degenerate cubelets, otherwise these can be restored in the dataset using the collate function, which will fill with zeros
            if cubelet.shape[0] < ysize or cubelet.shape[1] < xsize:
                cbs_indexes_degenerate.append(idx)
                if not keep_degenerate_cbs:
                    continue

            if masks is not None:
                # keep or not cubelets that are all nans
                mask_cubelet = masks[yslice, xslice]
                if missing_policy == "all":
                    if bool(mask_cubelet.all()):
                        cbs_indexes_missing.append(idx)
                        continue
                elif missing_policy == "any":
                    if bool(mask_cubelet.any()):
                        cbs_indexes_missing.append(idx)
                        continue
                else:
                    nmissing = mask_cubelet.sum()
                    total = mask_cubelet.shape[0] * mask_cubelet.shape[1]
                    missing_fraction = nmissing / total
                    if missing_fraction > missing_policy:
                        cbs_indexes_missing.append(idx)
                        continue

            cbs_slices.append([yslice, xslice])  # latlon
            cbs_indexes.append(idx)

            idx += 1

    assert len(cbs_slices) == len(cbs_indexes)

    return list(
        map(
            np.array,
            [cbs_indexes, cbs_indexes_missing, cbs_indexes_degenerate, cbs_slices],
        )
    )


def compute_cubelet_time_idxs(
    shape, tsize, tover, keep_degenerate_cbs=False, masks=None
):  # assume time,lat,lon
    time_size, lat_size, lon_size = shape

    idx = 0
    cbs_indexes, cbs_indexes_degenerate, cbs_slices = [], [], []

    for it in range(0, time_size, tsize - tover):
        tslice = slice(it, it + tsize)

        if len(range(time_size)[tslice]) < tsize:
            cbs_indexes_degenerate.append(idx)
            if not keep_degenerate_cbs:
                continue

        cbs_indexes.append(idx)
        cbs_slices.append(tslice)
        idx += 1

    return list(map(np.array, [cbs_indexes, cbs_indexes_degenerate, cbs_slices]))


def cbs_mapping_idx_slice(cbs_tuple_idxs, cbs_slices):
    mapping = {}
    for ic, islice in zip(cbs_tuple_idxs, cbs_slices):
        m = {"time": "", "lat": "", "lon": ""}
        sp_slice, t_slice = islice  # lat,lon,time
        tot_slice = (sp_slice[0], sp_slice[1], t_slice)  # T C H W
        m.update({"time": t_slice})
        m.update({"lat": sp_slice[0]})
        m.update({"lon": sp_slice[1]})
        mapping[ic] = m  # (sp_slice[0], sp_slice[1], t_slice)
    return mapping


def cbs_mapping_idx_slice_notime(cbs_tuple_idxs, cbs_slices):
    mapping = {}
    for ic, islice in zip(cbs_tuple_idxs, cbs_slices):
        m = {"lat": "", "lon": ""}
        # sp_slice= islice # lat,lon,time
        # tot_slice = (sp_slice[0], sp_slice[1], t_slice) # T C H W
        # m.update({"time":t_slice})
        m.update({"lat": islice[0]})
        m.update({"lon": islice[1]})
        mapping[ic] = m  # (sp_slice[0], sp_slice[1], t_slice)
    return mapping


def compute_cubelet_tuple_idxs(cbs_spatial_idxs, cbs_time_idxs):
    return list(itertools.product(*(cbs_spatial_idxs, cbs_time_idxs)))  # lat,lon,time


def compute_cubelet_slices(cbs_spatial_slices, cbs_time_slices):
    return list(
        itertools.product(*(cbs_spatial_slices, cbs_time_slices))
    )  # lat,lon,time


def get_unique_time_idxs(cbs_mapping_idxs):
    return np.unique([i[-1] for i in cbs_mapping_idxs.keys()]).tolist()


def get_unique_spatial_idxs(cbs_mapping_idxs):
    return np.unique([i[0] for i in cbs_mapping_idxs.keys()]).tolist()


def keep_valid(a, b):
    m1 = ~np.isnan(a)
    m2 = ~np.isnan(b)
    m3 = (m1) & (m2)
    return a[m3], b[m3]


def downsample_time(coords, frac):
    un = np.unique(coords[:, 0])
    for i, u in enumerate(un):
        idx = np.argwhere(coords[:, 0] == u)
        l = int(len(idx) * frac)
        idx2 = np.random.randint(idx[0], idx[-1] + 1, l)
        if i == 0:
            arr = coords[idx2]
        else:
            arr = np.concatenate([arr, coords[idx2]], axis=0)
    return arr


def downsample_space(coords, frac):
    un = np.unique(coords[:, 1])
    for i, u in enumerate(un):
        idx = np.argwhere(coords[:, 1] == u)
        l = int(len(idx) * frac)
        idx2 = np.random.randint(idx[0], idx[-1] + 1, l)
        if i == 0:
            arr = coords[idx2]
        else:
            arr = np.concatenate([arr, coords[idx2]], axis=0)
    return arr

def downsample_spacetime(coords, frac):
    l = int(len(coords) * frac)

    idx = np.random.randint(0, len(coords), l)
    
    
    return coords[idx]


def get_optimizer(model, cfg):
    if cfg.optimizer == "adam":
        opt = Adam(model.parameters(), lr=cfg.learning_rate)
    else:
        raise NotImplementedError

    return opt


def get_lr_scheduler(optimizer, cfg):
    try:
        config = cfg.get("lr_scheduler")
    except:
        config = cfg.lr_scheduler

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    if config is not None:
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config["mode"],
            factor=config["factor"],
            patience=config["patience"],
        )

    return lr_scheduler


def get_temporal_steps(steps):
    if steps == "all":
        selection = Ellipsis
    elif steps == 0:
        selection = -1
    else:
        selection = steps  
    return selection



def get_source_url_old(cfg):
    
    for k in cfg.data_source:
        if cfg.data_source.get(k, None) is not None:
            source = k
    if source == "file":
        url = f"{cfg.data_source['file']['data_dir']}/{cfg.data_source['file']['data_file']}"
    elif source == "s3":
        url = cfg.data_source['s3']['url']
    else:
        raise AttributeError
    return url

def get_source_url(cfg):
    
    xarray_kwargs = cfg.data_source.get("xarray_kwargs", {})
    
    data_sources = [k for k in cfg.data_source if k != "xarray_kwargs"]
    
    for k in data_sources:
        if cfg.data_source.get(k, None) is not None:
            source = k
    if source == "file":
        urls = {k:v for k,v in cfg.data_source['file'].items()}
    elif source == "s3":
        urls = {k:v for k,v in cfg.data_source['s3'].items()}
    else:
        raise AttributeError
    return urls, xarray_kwargs



def create_xarray_data(
    target,
    coords,
    output_shape,
    to_dataset_dim = "variable",
    crs = 4326,
) -> xr.Dataset | xr.DataArray:
    """
    output_shape and coords should have same dimensions (i.e. lat,lon,time,...)
    output_shape # (lat,lon,time)
    """

    if output_shape.get("variable") is None:
        output_shape["variable"] = target.shape[-1]
        
    reordered_out_shape = {}
    for v in ["lat", "lon", "time", "variable"]:
        if output_shape.get(v):
            reordered_out_shape[v] = output_shape[v]
        
    size = list(reordered_out_shape.values())
    y = target.reshape(*size)

    ds = xr.DataArray(y, dims=reordered_out_shape.keys(), coords=coords)
    
    if crs:
        ds = ds.rio.write_crs(crs)
        
    if to_dataset_dim:
        ds = ds.to_dataset(dim=to_dataset_dim)

    return ds

def rescale_target(ds, r, s):
    dsmin = ds.min("time")
    return ((ds - dsmin)/ (ds.max("time")- dsmin)) *(s-r) + r



def unnest(l):
    out = []
    for i in l:
        out.append([*i[0],i[1]])
    return np.array(out)