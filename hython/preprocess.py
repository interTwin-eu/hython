import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import xarray as xr

from typing import List
from typing import Union, Any 
from dask.array.core import Array as DaskArray

from dask.array import expand_dims, nanmean, nanstd, nanmin, nanmax

def reshape(
            dynamic: xr.Dataset,
            static: xr.Dataset,
            target: xr.Dataset,
            return_type = "xarray"
              ) -> List[DaskArray] | List[xr.DataArray]:

    # reshape 
    Xd = ( dynamic
        .to_dataarray(dim="feat") # cast
        .stack(gridcell= ["lat","lon"]) # stack 
        .transpose("gridcell","time","feat") 
        )
    print("dynamic: ", Xd.shape, " => (GRIDCELL, TIME, FEATURE)")
    
    Xs = ( static
    .drop_vars("spatial_ref")
    .to_dataarray(dim="feat")
    .stack(gridcell= ["lat","lon"])
    .transpose("gridcell","feat")
    )
    print("static: ", Xs.shape, " => (GRIDCELL, FEATURE)")

    Y = ( target
        .to_dataarray(dim="feat")
        .stack(gridcell= ["lat","lon"])
        .transpose("gridcell","time", "feat")
        )
    print("target: ", Y.shape, " => (GRIDCELL, TIME, TARGET)")     

    if return_type == "xarray":
        return Xd, Xs, Y
    if return_type == "dask":
        return Xd.data, Xs.data, Y.data
    if return_type == "numpy":
        return Xd.compute().values, Xs.compute().values, Y.compute().values
        

def scale(a, how, axis, m1, m2):
    
    if how == 'standard':
        if m1 is None or m2 is None:
            
            m1, m2 = nanmean(a, axis=axis), nanstd(a, axis=axis)
            
            return (a - expand_dims(m1, axis = axis) )/ expand_dims(m2, axis = axis), m1, m2
        else:
            return (a - expand_dims(m1, axis = axis))/ expand_dims(m2, axis = axis)
            
    elif how == 'minmax':
        if m1 is None or m2 is None:
            m1, m2 = nanmin(a, axis=axis), nanmax(a, axis=axis)
            
            den = m2 - m1 
                
            return (a - m1)/den, m1, m2
        else:

            den = m2 - m1
            
            return (a - m1)/den
                
def apply_normalization(a, type = "time", how='standard', m1=None, m2=None):
    """Assumes array of 
    dynamic: (gridcell, time, dimension)
    static: (gridcell, dimension)

    Parameters
    ----------
    a : 
        _description_
    type : str, optional
        , by default "space"
    how : str, optional
        _description_, by default 'standard'
    m1 : _type_, optional
        _description_, by default None
    m2 : _type_, optional
        _description_, by default None
    """
    if type == "time":
        return scale(a, how = how, axis = 1, m1 = m1, m2 = m2)
    elif type == "space": 
        return scale(a, how = how, axis = 0, m1 = m1,  m2 = m2)
    elif type == "spacetime":
         return scale(a, how = how, axis = (0, 1), m1 = m1, m2 = m2)
    else:
        raise NotImplementedError(f"Type {how} not implemented")



# TODO add splitting by dates
def split_time(a, slice):
    return a[:,:,slice]


def split_space(a, sampler):
    return sampler.sampling(a)



def split_time_and_space(a, 
                         validation_type, 
                         temporal_train_slice = None, 
                         temporal_val_slice = None, 
                         spatial_train_sampler = None,
                         spatial_val_sampler = None):
    # split time
    if validation_type == "space" or validation_type == "spacetime":
        a_train = split_space(a, spatial_train_sampler)[0]
        a_val = split_space(a, spatial_val_sampler)[0]
    
    if validation_type == "time":
        a_train = split_time(a, temporal_train_slice)
        a_val = split_time(a, temporal_val_slice)

    if validation_type == "spacetime":
        a_train = split_time(a_train, temporal_train_slice)
        a_val = split_time(a_val, temporal_val_slice)   

    orig_shape = a.shape
    train_shape = a_train.shape 
    val_shape = a_val.shape
    print(f"""Approach {validation_type}: \n
          Original dataset: (lat {orig_shape[0]}, lon {orig_shape[1]} , time {orig_shape[2]} , feature {orig_shape[3]}) \n
          Train dataset: (lat {train_shape[0]}, lon {train_shape[1]} , time {train_shape[2]} , feature {train_shape[3]}) \n
          Validation dataset: (lat {val_shape[0]}, lon {val_shape[1]} , time {val_shape[2]} , feature {val_shape[3]})""")
        
    return a_train, a_val
    
