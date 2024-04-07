import numpy as np
import xarray as xr
import torch
import cf_xarray as cfxr

import os, random

from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates
import zarr
from typing import Any
from numpy.typing import NDArray
from dask.array.core import Array as DaskArray

def load(surrogate_input_path, wflow_model, files = ["Xd", "Xs", "Y"]):
    loaded = np.load(surrogate_input_path / f"{wflow_model}.npz")
    return [loaded[f] for f in files]

def missing_location_idx(grid: np.ndarray |  xr.DataArray | xr.Dataset,
                           missing: Any = np.nan) -> NDArray | list:
    """Returns the indices corresponding to missing values

    Args:
        grid (np.ndarray | xr.DataArray | xr.Dataset): _description_
        missing (Any, optional): _description_. Defaults to np.nan.

    Returns:
        np.array | list: _description_
    """
    
    if isinstance(grid, np.ndarray) or isinstance(grid, torch.Tensor):
        shape = grid.shape
    elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
        shape = grid.gridcell
    else:
        pass

    if isinstance(grid, np.ndarray) or isinstance(grid, torch.Tensor):
        
        location_idx = np.isnan(grid).any(axis = -1)

    elif isinstance(grid, xr.DataArray) or isinstance(grid, xr.Dataset):
        pass
    else:
        pass

    return location_idx # (location, dims)



def predict(Xd, Xs, model, batch_size, device):
    model = model.to(device)
    arr = []
    for i in range(0,Xd.shape[0], batch_size):
        
        d = torch.Tensor(Xd[i:(i+batch_size)]).to(device)
        
        s = torch.Tensor(Xs[i:(i+batch_size)]).to(device)
        arr.append(
            model(d, s).detach().cpu().numpy()
            )
    return np.vstack(arr)


def to_xr(arr, coords, dims = ["lat", "lon", "time"]):
    return xr.DataArray(arr,
                        dims = dims,
                           coords = coords)

def reshape_to_2Dspatial(a, lat_size, lon_size, time_size, feat_size, coords= None):
    tmp = a.reshape(lat_size, lon_size, time_size , feat_size)
    return tmp


def reconstruct_from_missing(a: NDArray, original_shape: tuple, missing_location_idx: NDArray) -> NDArray:
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
         *(original_shape[1:] if len(original_shape) > 2 else [original_shape[1]])
         ),
        np.nan
         )
    print(fill.shape)
    
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

def write_to_zarr(arr: DaskArray | xr.DataArray,
                  url, 
                  group = None, 
                  storage_options = {}, 
                  overwrite = True, 
                  chunks="auto", 
                  clear_zarr_storage = False,
                  append_on_time = False,
                  time_chunk_size = 200,
                  multi_index = None):
    
    if isinstance(arr, DaskArray):
        arr = arr.rechunk(chunks = chunks)
        arr.to_zarr(url=url, storage_options=storage_options, overwrite=overwrite, component=group)
        
    if isinstance(arr, xr.DataArray):
        if overwrite:
            overwrite="w"
        else:
            overwrite="r"
        if chunks:
            arr = arr.chunk(chunks=chunks)
        shape = arr.shape
        if multi_index:
            arr = arr.to_dataset(name=group)
            arr = cfxr.encode_multi_index_as_compress(arr, multi_index)

        if append_on_time:
            fs_store = zarr.storage.FSStore(url, storage_options=storage_options, mode=overwrite)
            
            if clear_zarr_storage:
                fs_store.clear()
            
            # initialize
            init = arr.isel(time=slice(0,time_chunk_size)).persist()
            init[group].attrs.clear()
            init.to_zarr(fs_store, consolidated = True, group=group)
            for t in range(time_chunk_size, shape[1], time_chunk_size): # append time
                arr.isel(time=slice(t,t+time_chunk_size)).to_zarr(fs_store, append_dim="time", consolidated=True, group=group)
        else:
            arr.to_zarr(store=url, storage_options=storage_options, mode=overwrite, group=group)
        
def read_from_zarr(url, group = None, multi_index = None, engine = "xarray"):
    if engine == "xarray":
        ds = xr.open_dataset(url, group=group, engine="zarr")
        if multi_index:
            ds = cfxr.decode_compress_to_multi_index(ds, multi_index)
        return ds
    

def prepare_for_plotting(y_target: NDArray, y_pred: NDArray, shape: tuple[int], coords: DataArrayCoordinates | DatasetCoordinates):
    
    lat, lon, time = shape
    n_feat = y_target.shape[-1]
    
    y = reshape_to_2Dspatial(
            y_target,
            lat,
            lon,
            time,
            n_feat)
     
    yhat = reshape_to_2Dspatial(
            y_pred,
            lat,
            lon,
            time,
            n_feat)
    
    y = to_xr(y[...,0], coords = coords)
    yhat = to_xr(yhat[...,0], coords = coords)
    
    return y, yhat

def get_sampler_config(surrogate_experiment_name):
    if surrogate_experiment_name == "s0001": # 0.1 %
        print("0.1 %")
        intervals = [12, 12]
        val_origin = [1 ,1 ]
        train_origin = [0, 0]
    elif surrogate_experiment_name == "s0010":  # 1 %
        print("1 %")
        intervals = [4, 4]
        val_origin = [1, 1]
        train_origin = [0, 0]
    elif surrogate_experiment_name == "s00001":  # 0.01 %
        print("0.01 %")
        intervals = [36, 36]
        val_origin = [1, 1]
        train_origin = [0, 0]
    elif surrogate_experiment_name == "s000001":  # 0.001 %
        print("0.001 %")
        intervals = [108, 108]
        val_origin = [64, 64]
        train_origin = [0, 0]
    else:
        raise Exception("experiment not found")
        
    return intervals, val_origin, train_origin

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
