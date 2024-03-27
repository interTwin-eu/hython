from itwinai.components import DataGetter, monitor_exec
from typing import List, Dict
from pathlib import Path
import torch

#import gdown
import numpy as np
import xarray as xr

from hython.preprocess import reshape, apply_normalization
from hython.datasets.datasets import LSTMDataset
from hython.sampler import RegularIntervalSampler, DataLoaderSpatialSampler


class LSTMDataGetter(DataGetter):
    def __init__(
        self,
        wd: str, 
        wflow_model: str,
        fn_dynamic_forcings: str, 
        fn_wflow_static_params: str, 
        fn_target: str, 
        dynamic_names: list, 
        static_names : list, 
        target_names : list, 
        timeslice    : list, 
        intervals    : list,
        train_origin : list, 
        valid_origin : list, 
    ):
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))

        self.wd = wd
        self.wflow_model = wflow_model
        self.fn_dynamic_forcings    = fn_dynamic_forcings
        self.fn_wflow_static_params = fn_wflow_static_params
        self.fn_target     = fn_target
        self.dynamic_names =  dynamic_names
        self.static_names  =  static_names 
        self.target_names  =  target_names 
        self.timeslice     =  slice(timeslice[0], timeslice[1])
        self.intervals     = tuple(intervals)
        self.train_origin  = tuple(train_origin)
        self.valid_origin  = tuple(valid_origin)


    @monitor_exec
    def execute(self):
        
        p = Path(self.wd) / self.wflow_model

        forcings = xr.open_dataset( p / self.fn_dynamic_forcings)
        params = xr.open_dataset(p / self.fn_wflow_static_params ).sel(layer=1)
        targets = xr.open_dataset(p / "run_default" / self.fn_target).sel(layer=1).isel(lat=slice(None, None, -1))

        try:
            forcings = forcings.rename({"latitude":"lat", "longitude":"lon"})
            params = params.rename({"latitude":"lat", "longitude":"lon"})
        except:
            pass

        wflow_dem = params.wflow_dem
        wflow_lakes = params.wflow_lakeareas.values

        mask_lakes = (wflow_lakes > 0).astype(np.bool_)

        if self.timeslice:
            forcings = forcings.sel(time=self.timeslice)
            targets = targets.sel(time=self.timeslice)

        forcings = forcings[self.dynamic_names]
        params = params[self.static_names]
        targets = targets[self.target_names] 

        Xd, Xs, Y  = reshape(
                    forcings, 
                    params, 
                    targets,
                    return_type="dask"
                    ) 
        
        #define data mask
        missing_mask = np.isnan(params[self.static_names[0]]) | mask_lakes

        spatial_train_sampler = RegularIntervalSampler(intervals = self.intervals, origin = self.train_origin)
        spatial_val_sampler = RegularIntervalSampler(intervals = self.intervals, origin = self.valid_origin) 

        sampler_train_meta = spatial_train_sampler.sampling_idx(wflow_dem, missing_mask)
        sampler_val_meta = spatial_val_sampler.sampling_idx(wflow_dem, missing_mask)

        _, d_m, d_std = apply_normalization(Xd[sampler_train_meta.idx_sampled_1d_nomissing], type = "spacetime", how ='standard')
        _, s_m, s_std = apply_normalization(Xs[sampler_train_meta.idx_sampled_1d_nomissing], type = "space", how ='standard')
        _, y_m, y_std = apply_normalization(Y[sampler_train_meta.idx_sampled_1d_nomissing], type = "spacetime", how ='standard')

        Xd = apply_normalization(Xd, type="spacetime", how="standard", m1 = d_m, m2 = d_std)
        Xs = apply_normalization(Xs, type="space", how="standard",  m1 = s_m, m2 = s_std)
        Y = apply_normalization(Y, type="spacetime",how="standard", m1 = y_m, m2 = y_std)

        Xd = Xd.compute()
        Y = Y.compute()
        Xs = Xs.compute()

        Xs = torch.Tensor(Xs)
        Xd = torch.Tensor(Xd)
        Y = torch.Tensor(Y)

        dataset = LSTMDataset(Xd, Y, Xs)

        train_sampler = DataLoaderSpatialSampler(dataset, num_samples=100, sampling_indices = sampler_train_meta.idx_sampled_1d_nomissing.tolist())
        valid_sampler = DataLoaderSpatialSampler(dataset, num_samples=100, sampling_indices = sampler_val_meta.idx_sampled_1d_nomissing.tolist())

        return dataset, train_sampler, valid_sampler