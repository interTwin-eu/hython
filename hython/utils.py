import os, random
import numpy as np
import xarray as xr
import torch

from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

from typing import Any
from numpy.typing import NDArray
from dask.array.core import Array as DaskArray
import itertools

def generate_model_name(surr_model_prefix, experiment, target_names, hidden_size, seed):
    TARGET_INITIALS = "".join([i[0].capitalize() for i in target_names])
    return f"{surr_model_prefix}_{experiment}_v{TARGET_INITIALS}_h{hidden_size}_s{seed}.pt"

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

def reshape_to_2Dspatial(a, lat_size, lon_size, time_size, feat_size, coords=None):
    tmp = a.reshape(lat_size, lon_size, time_size, feat_size)
    return tmp

def prepare_for_plotting(
    y_target: NDArray,
    y_pred: NDArray,
    shape: tuple[int],
    coords: DataArrayCoordinates | DatasetCoordinates,
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

def compute_cubelet_spatial_idxs(shape, xsize, ysize, xover, yover, keep_degenerate_cbs=False, masks = None,
                                 missing_policy = "all"): # assume time,lat,lon

    time_size,lat_size,lon_size = shape

    # compute 
    space_idx = compute_grid_indices(shape = (lat_size,lon_size))
    
    idx = 0
    cbs_indexes, cbs_indexes_missing, cbs_indexes_degenerate, cbs_slices = [],[],[],[]
    
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
                    if mask_cubelet.all().item(0):
                        cbs_indexes_missing.append(idx)
                        continue
                elif missing_policy == "any":
                    if mask_cubelet.any().item(0):
                        cbs_indexes_missing.append(idx)
                        continue
                else:
                    nmissing = mask_cubelet.sum()
                    total = mask_cubelet.shape[0] * mask_cubelet.shape[1]
                    missing_fraction = nmissing/total
                    if missing_fraction > missing_policy:
                        cbs_indexes_missing.append(idx)
                        continue
                
            cbs_slices.append([yslice, xslice]) # latlon
            cbs_indexes.append(idx)
                
            idx += 1

    assert len(cbs_slices) == len(cbs_indexes)
    
    return list(map(np.array, [cbs_indexes,cbs_indexes_missing, cbs_indexes_degenerate, cbs_slices]))

def compute_cubelet_time_idxs(shape, tsize, tover, keep_degenerate_cbs = False, masks = None): # assume time,lat,lon

    time_size,lat_size,lon_size = shape

    idx = 0
    cbs_indexes, cbs_indexes_degenerate, cbs_slices = [],[], []

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
        m = {"time":"", "lat":"", "lon":""}
        sp_slice, t_slice = islice # lat,lon,time
        tot_slice = (sp_slice[0], sp_slice[1], t_slice) # T C H W
        m.update({"time":t_slice})
        m.update({"lat":sp_slice[0]})
        m.update({"lon":sp_slice[1]})
        mapping[ic] = m # (sp_slice[0], sp_slice[1], t_slice)    
    return mapping

def compute_cubelet_tuple_idxs(cbs_spatial_idxs, cbs_time_idxs):
     return list(itertools.product(*(cbs_spatial_idxs, cbs_time_idxs))) # lat,lon,time

def compute_cubelet_slices(cbs_spatial_slices, cbs_time_slices):
     return list(itertools.product(*(cbs_spatial_slices, cbs_time_slices))) # lat,lon,time

def get_unique_time_idxs(cbs_mapping_idxs):
    return np.unique([i[-1] for i in cbs_mapping_idxs.keys()]).tolist()
def get_unique_spatial_idxs(cbs_mapping_idxs):
    return np.unique([i[0] for i in cbs_mapping_idxs.keys()]).tolist()



