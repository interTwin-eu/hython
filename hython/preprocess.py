import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import xarray as xr
from hython.sampler import AbstractSampler

from typing import List
from typing import Union, Any 


def preprocess(
            dynamic: xr.Dataset,
            static: xr.Dataset,
            target: xr.Dataset,
            dynamic_name: List,
            static_name: List, 
            target_name: List,
            sampler: AbstractSampler = None
              ) -> List[np.ndarray]:

    DIMS = {
        "orig": [ len(dynamic["lat"]), len(dynamic["lon"]), len(dynamic["time"])  ],
    }

    META = {"dyn":"", "static":"", "target":""}

    # select
    dyn_sel = dynamic[dynamic_name]
    static_sel = static[static_name]
    target_sel = target[target_name]

    # sampling

    if sampler:
        dyn_sel, dyn_sampler_meta = sampler.sampling(dyn_sel.transpose("lat", "lon", "time"))
        static_sel, static_sampler_meta = sampler.sampling(static_sel.transpose("lat", "lon"))
        target_sel, target_sampler_meta = sampler.sampling(target_sel.transpose("lat", "lon", "time"))

        DIMS["sampled_dims"] =  [ len(dyn_sel["lat"]), len(dyn_sel["lon"]), len(dyn_sel["time"])  ]

        print("sampling reduced dims (lat, lon): from ", DIMS["orig"][:2], " to ", DIMS["sampled_dims"][:2] )

        META.update({"dyn":dyn_sampler_meta,"static":static_sampler_meta, "target":target_sampler_meta})

    # train_test split 


    # reshape 
    Xd = ( dyn_sel
        .to_dataarray(dim="feat") # cast
        .stack(cell= ["lat","lon"]) # stack 
        .transpose("cell","time","feat") 
        )
    print("dynamic: ", Xd.shape, " => (GRIDCELL, TIME, FEATURE)")
    
    Xs = ( static_sel
    .drop_vars("spatial_ref")
    .to_dataarray(dim="feat")
    .stack(cell= ["lat","lon"])
    .transpose("cell","feat")
    )
    print("static: ", Xs.shape, " => (GRIDCELL, FEATURE)")

    Y = ( target_sel
        .to_dataarray(dim="feat")
        .stack(cell= ["lat","lon"])
        .transpose("cell","time", "feat")
        )
    print("target: ", Y.shape, " => (GRIDCELL, TIME, TARGET)")     



    return Xd.compute().values,Xs.compute().values, Y.compute().values, DIMS, META

def scale(a, how, axis, m1, m2):
    if how == 'standard':
        if m1 is None or m2 is None:
            m1, m2 = np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)
            
            m2[m2 == 0] = 1
            
            return (a - np.expand_dims(m1, axis = axis) )/ np.expand_dims(m2, axis = axis), m1, m2
        else:
            return (a - np.expand_dims(m2, axis = axis))/np.expand_dims(m2, axis = axis)
    elif how == 'minmax':
        if m1 is None or m2 is None:
            m1, m2 = np.nanmin(a, axis=axis), np.nanmax(a, axis=axis)
            return (a - m1)/(m2 - m1), m1, m2
        else:
            return (a - m1)/(m2 - m1)
                
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
    
