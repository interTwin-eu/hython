import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import dask
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
            sampler: AbstractSampler = None,
            return_sampler_meta: bool = False
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


def apply_missing_policy(Xd: np.ndarray, Xs: np.ndarray, Y: np.ndarray, 
                         policy_missing: dict[dict] = None) -> np.ndarray:
    
    if isinstance(Xd, np.ndarray):
        print("Applying missing value policy...")
        if ps := policy_missing.get("static"):
            if (psr := ps.get("replace")) is not None:
                Xs = np.where(np.isnan(Xs), psr, Xs)
        if pd := policy_missing.get("dynamic"):
            if (pdr := pd.get("replace")) is not None:
                Xd = np.where(np.isnan(Xd), pdr, Xd)
        if pt := policy_missing.get("target"):
            if (ptr := pt.get("replace")) is not None:
                Y = np.where(np.isnan(Y), ptr, Y)
        print("...done")
    elif isinstance(Xd, dask.array.core.Array):
        from dask.array import isnan, where
        if ps := policy_missing.get("static"):
            if (psr := ps.get("replace")) is not None:
                Xs = where(isnan(Xs), psr, Xs)
        if pd := policy_missing.get("dynamic"):
            if (pdr := pd.get("replace")) is not None:
                Xd = where(isnan(Xd), pdr, Xd)
        if pt := policy_missing.get("target"):
            if (ptr := pt.get("replace")) is not None:
                Y = where(isnan(Y), ptr, Y)
        print("...done")
    else:
        print("Applying missing value policy...")
        if ps := policy_missing.get("static"):
            if (psr := ps.get("replace")) is not None:
                Xs = Xs.where(~isnan(Xs), psr)
        if pd := policy_missing.get("dynamic"):
            if (pdr := pd.get("replace")) is not None:
                Xd = Xd.where(~isnan(Xd), pdr)
        if pt := policy_missing.get("target"):
            if (ptr := pt.get("replace")) is not None:
                Y = Y.where(~isnan(Y), ptr)
        print("...done")
    return Xd, Xs, Y



def apply_normalization(a, type = "time", how='standard', m=None, std=None):
    """Assumes array of 
    dynamic: (gridcell, time, dimension)
    static: (gridcell, dimension)

    Parameters
    ----------
    x : _type_
        _description_
    type : str, optional
        , by default "space"
    how : str, optional
        _description_, by default 'standard'
    m : _type_, optional
        _description_, by default None
    std : _type_, optional
        _description_, by default None
    """

    def scale(a, how, axis, m, std):
        if how == 'standard':
            if m is None or std is None:
                m, std = np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)
                
                std[std == 0] = 1
                
                return (a - np.expand_dims(m, axis = axis) )/ np.expand_dims(std, axis = axis), m, std
            else:
                return (a - np.expand_dims(m, axis = axis))/np.expand_dims(std, axis = axis)
        elif how == 'minmax':
            mmin, mmax = np.nanmin(a, axis=axis), np.nanmax(a, axis=axis)
            return (a - mmin)/(mmax- mmin), mmin, mmax

    if type == "time":
        return scale(a, how = how, axis = 1, m = m, std = std)
    elif type == "space": 
        return scale(a, how = how, axis = 0, m = m, std = std)
    else:
         return scale(a, how = how, axis = (0, 1), m = m, std = std)




