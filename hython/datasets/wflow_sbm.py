"""Wflow_sbm emulators"""
from . import *
import itertools
import logging
from hython.preprocessor import Preprocessor

LOGGER = logging.getLogger(__name__)


class WflowSBM_HPC(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train"
    ):
        self.scaler = scaler
        self.seq_len = cfg.seq_length
        self.cfg = self.validate_config(cfg)

        self.preprocessor = Preprocessor(cfg)

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        self.target_has_missing_dates = self.cfg.get("target_has_missing_dates", False)

        urls, xarray_kwargs = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")
        
        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs).sel(time=self.period_range)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto", **xarray_kwargs)

        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)] # list comprehension handle omegaconf lists
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_dynamic[self.to_list(cfg.target_variables)]
        
        # subset dynamic inputs to the target timestep available
        if self.target_has_missing_dates:
            self.xd = self.xd.sel(time=self.y.time)

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()

        # == DATASET INDICES AND MASKING

        if self.cfg.mask_variables is not None and self.period != "test":
            # During training and validation remove cells marked as mask.
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            self.cell_coords = np.argwhere(~self.mask.values)
        elif self.period == "test": 
            # No masking during testing, however computing mask is still useful.
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            shape = list(self.xs.dims.values())
            self.cell_coords =  np.argwhere(np.ones(shape).astype(bool))

        # Compute cell (spatial) index 
        self.cell_linear_index  = np.arange(0, len(self.cell_coords ), 1)
        
        # Compute sequence (temporal) index
        # Each cell has a time series of equal length, so the sequence index is the same for every cell
        if self.period == "test":
            self.time_index = np.arange(0, len(self.xd.time.values), 1)
        else:
            self.time_index = np.arange(0, len(self.xd.time.values) - self.seq_len, 1)
        
        # == DOWNSAMPLING

        # Downsample spatial and temporal indices based on rule
        if self.downsampler is not None:
            self.cell_linear_index , self.time_index = self.downsampler.sampling_idx([self.cell_linear_index , self.time_index])

        if self.period == "test":
            self.spacetime_index = self.cell_linear_index
        else:
            self.spacetime_index = list(itertools.product(*[
                                                        self.cell_coords[self.cell_linear_index ].tolist(), 
                                                        self.time_index.tolist() 
                                                        ]))  

            self.spacetime_index = unnest(self.spacetime_index)

        # == SOME USEFUL PARAMETERS
        self.lat_size = len(self.xd.lat)
        self.lon_size = len(self.xd.lon)
        self.time_size = len(self.xd.time)
        self.dynamic_coords = self.xd.coords
        self.static_coords = self.xs.coords


        #  === PREPROCESS/TRANSFORM VARIABLES

        if self.cfg.get("preprocessor") is not None:
            self.xs = self.preprocessor.process(self.xs, "static_inputs")
            self.xd = self.preprocessor.process(self.xd, "dynamic_inputs")
            self.y = self.preprocessor.process(self.y, "target_variables")
        
        # == SCALING 

        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("lat","lon", "time")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat","lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("lat", "lon", "time")
        )

        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")

        self.y = self.scaler.transform(self.y, "target_variables")

        self.xs = self.scaler.transform(self.xs, "static_inputs")
        
        # == WRITE SCALING STATS

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")


        # Pre-compute static data once
        self.static_tensor = torch.tensor(self.xs.to_array().values).float()
        
        # Pre-compute dynamic data shapes once
        self.dynamic_shape = self.xd[self.cfg.dynamic_inputs[0]].shape
        self.target_shape = self.y[self.cfg.target_variables[0]].shape
    

        # Convert to tensors once during initialization
        self.xd = self.xd.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        ).transpose("time", "feat", "lat", "lon").astype("float32")
        self.y = self.y.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        ).transpose("time", "feat", "lat", "lon").astype("float32")
        self.xs = self.xs.to_stacked_array(
            new_dim="feat", sample_dims=["lat", "lon"]
        ).transpose("feat", "lat", "lon").astype("float32")

        # Pre-process once
        if not self.cfg.data_lazy_load:  # Only if we're not doing lazy loading
            # Convert xarray to pre-processed tensors
            self.xd = torch.from_numpy(
                self.xd.transpose("time", "feat", "lat", "lon").values
            )
            self.y = torch.from_numpy(
                self.y.transpose("time", "feat", "lat", "lon").values
            )
            self.xs = torch.from_numpy(
                self.xs.transpose("feat", "lat", "lon").values
            )

    def __len__(self):
        return len(self.spacetime_index)

    def __getitem__(self, index):
        
        if self.period == "test":
            idx_lat, idx_lon = self.cell_coords[index]

            xd = self.xd[:,:,idx_lat, idx_lon]

            y = self.y[:,:,idx_lat, idx_lon]
            
            xs = self.xs[:, idx_lat, idx_lon]
        else:
            idx_lat, idx_lon, idx_time = self.spacetime_index[index]

            xd = self.xd[idx_time:idx_time + self.seq_len,:,idx_lat, idx_lon]

            y = self.y[idx_time:idx_time + self.seq_len,:,idx_lat, idx_lon]
            
            xs = self.xs[:, idx_lat, idx_lon]

        return {"xd": xd, "xs": xs, "y": y}

class WflowSBM(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler

        self.cfg = self.validate_config(cfg)

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        self.target_has_missing_dates = self.cfg.get("target_has_missing_dates", False)

        urls, xarray_kwargs = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")

        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs).sel(time=self.period_range)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto", **xarray_kwargs)
        
        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)] # list comprehension handle omegaconf lists
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_dynamic[self.to_list(cfg.target_variables)]

        # subset dynamic inputs to the target timestep available
        if self.target_has_missing_dates:
            self.xd = self.xd.sel(time=self.y.time)

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()

        # == DATASET INDICES AND MASKING

        if self.cfg.mask_variables is not None and self.period != "test":
            # During training and validation remove cells marked as mask.
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            self.coords = np.argwhere(~self.mask.values)
        elif self.period == "test": 
            # No masking during testing, however computing mask is still useful.
            self.mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            shape = list(self.xs.dims.values())
            self.coords =  np.argwhere(np.ones(shape).astype(bool))

        # Compute cell (spatial) index 
        self.cell_index = np.arange(0, len(self.coords), 1)
        
        # Compute sequence (temporal) index
        # Each cell has a time series of equal length, so the sequence index is the same for every cell
        if self.cfg.downsampling_temporal_dynamic or self.period == "test":
            self.time_index = np.arange(0, len(self.xd.time.values), 1)
        else:
            self.time_index = np.arange(self.seq_len, len(self.xd.time.values), 1)
        
        # == DOWNSAMPLING

        # Downsample spatial and temporal indices based on rule
        if self.downsampler is not None:
            self.cell_index, self.time_index = self.downsampler.sampling_idx([self.cell_index, self.time_index])

        # Generate dataset samples
        if self.cfg.downsampling_temporal_dynamic:
            # Only downsample the spatial index, the time index to downsample the sequences, is generated at runtime.
            # Therefore the dataset samples are sequences of max time series length. 
            self.coord_samples = self.coords[self.cell_index]
        else:
            # The dataset samples are the combination of cell and time indices
            self.coord_samples = list(itertools.product(*(self.coords[self.cell_index].tolist(), self.time_index.tolist() )))  
        
        # == SCALING 

        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("lat","lon", "time")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat","lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("lat", "lon", "time")
        )

        if not self.scale_ontraining:
            self.xd = self.scaler.transform(self.xd, "dynamic_inputs")

            self.y = self.scaler.transform(self.y, "target_variables")

            if self.scaling_static_range is not None:
                LOGGER.info(f"Scaling static inputs with {self.scaling_static_range}")   
                scaling_static_reordered = {
                    k: self.cfg.scaling_static_range[k]
                    for k in self.cfg.static_inputs
                    if k in self.cfg.scaling_static_range
                }

                self.static_scale, self.static_center = self.get_scaling_parameter(
                    scaling_static_reordered, self.cfg.static_inputs, output_type="xarray"
                )
                self.xs = self.scaler.transform_custom_range(
                    self.xs, self.static_scale, self.static_center
                )
            else:
                self.xs = self.scaler.transform(self.xs, "static_inputs")
        else:
            # these will be used in the getitem by the scaler.transform_custom_range
            scaling_static_reordered = {
                k: self.cfg.scaling_static_range[k]
                for k in self.cfg.static_inputs
                if k in self.cfg.scaling_static_range
            }

            self.static_scale, self.static_center = self.get_scaling_parameter(
                scaling_static_reordered, self.cfg.static_inputs
            )
        
        # == WRITE SCALING STATS

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")

    def __len__(self):
        return len((range(len(self.coord_samples))))

    def __getitem__(self, index):
        
        if self.scale_ontraining:
            pass
            #TODO: implement 
            # xd = torch.tensor(
            #     self.scaler.transform(self.xd[item_index], "dynamic_inputs").values
            # ).float()

            # y = torch.tensor(
            #     self.scaler.transform(self.y[item_index], "target_variables").values
            # ).float()

            # if self.scaling_static_range is not None:
            #     xs = torch.tensor(
            #         self.scaler.transform_custom_range(
            #             self.xs[item_index],
            #             "static_inputs",
            #             self.static_scale,
            #             self.static_center,
            #         ).values
            #     ).float()
            # else:
            #     xs = torch.tensor(
            #         self.scaler.transform(self.xs[item_index], "static_inputs").values
            #     ).float()

        else:
            if self.cfg.downsampling_temporal_dynamic:
                lat, lon = self.coord_samples[index]

                ds_pixel_dynamic = self.xd.isel(lat=lat, lon=lon) # lat, lon, time -> time
                ds_pixel_target = self.y.isel(lat=lat, lon=lon)
                ds_pixel_static = self.xs.isel(lat=lat, lon=lon)

                ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
                ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature

                ds_pixel_static = ds_pixel_static.to_array()
                
                xd  = torch.tensor(ds_pixel_dynamic.values).float()
                xs = torch.tensor(ds_pixel_static.values).float()
                y = torch.tensor(ds_pixel_target.values).float()
            else:
                idx_cell, idx_time = self.coord_samples[index]

                idx_lat, idx_lon = idx_cell
                # TODO: check
                ds_pixel_dynamic = self.xd.isel(lat=idx_lat, 
                                                        lon=idx_lon, 
                                                        time=slice(idx_time - self.seq_len + 1, idx_time + 1)) # lat, lon, time -> time

                ds_pixel_target = self.y.isel(lat=idx_lat, 
                                                        lon=idx_lon, 
                                                        time=slice(idx_time - self.seq_len + 1, idx_time + 1)) 
                
                ds_pixel_static = self.xs.isel(lat=idx_lat, lon=idx_lon)
        
                ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
                ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature
                ds_pixel_static = ds_pixel_static.to_array()
                
                xd  = torch.tensor(ds_pixel_dynamic.values).float()
                xs = torch.tensor(ds_pixel_static.values).float()
                y = torch.tensor(ds_pixel_target.values).float()

        return {"xd": xd, "xs": xs, "y": y}

class WflowSBMCal(BaseDataset):
    """Dataset returns sequences with length equal to calibration period
    """

    def __init__(self, cfg, scaler, is_train=True, period="train", scale_ontraining=False):
        super().__init__()
        self.cfg = self.validate_config(cfg)
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])
        self.scaling_static_range = self.cfg.get("scaling_static_range")

        self.target_has_missing_dates = self.cfg.get("target_has_missing_dates")
        
        urls, xarray_kwargs = get_source_url(cfg)

        # load datasets
        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto", **xarray_kwargs)
        data_target = read_from_zarr(url=urls["target_variables"], chunks="auto", **xarray_kwargs)
        
        # select 
        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)].sel(time=self.period_range) # list comprehension handle omegaconf lists
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_target[self.to_list(cfg.target_variables)].sel(time=self.period_range)

        # subset dynamic inputs to the target timestep available
        
        if self.target_has_missing_dates is not None:
            self.xd = self.xd.sel(time=self.y.time)
            

        if cfg.scaling_rescale_target is not None:
            print("Rescaling target")
            # updates target to statistics of vwc
            self.rescale_target(urls, data_dynamic, data_target, xarray_kwargs)

        # TODO: ensure they are all float32
        # head_layer mask
        head_mask = read_from_zarr(url=urls["mask_variables"], chunks="auto", **xarray_kwargs)
        self.head_mask = head_mask[self.to_list(self.cfg.mask_variables)].to_array().any("variable")

        # == DATASET INDICES AND MASKING
        # target mask, observation
        if urls.get("target_variables_mask", None):
            target_mask = read_from_zarr(url=urls["target_variables_mask"], chunks="auto",**xarray_kwargs).sel(time=self.period_range)
            sel_target_mask = self.to_list(self.cfg.target_variables_mask)[0] if isinstance(self.to_list(self.cfg.target_variables_mask), list) else self.to_list(self.cfg.target_variables_mask)
            self.target_mask = target_mask[sel_target_mask]
            self.target_mask = self.target_mask.resample({"time":"1D"}).max().astype(bool)
            self.target_mask = self.target_mask.isnull().sum("time") > self.cfg.min_sample_target     
        else:
            self.target_mask = self.y.isnull().all("time")[self.to_list(cfg.target_variables)[0]] #> self.cfg.min_sample_target   

        # static mask, predictors
        if urls.get("static_inputs_mask", None):
            self.static_mask = read_from_zarr(url=urls["static_inputs_mask"], chunks="auto", **xarray_kwargs)[self.to_list(self.cfg.static_inputs_mask)[0]]
        else:
            self.static_mask = self.xs.isnull()[self.to_list(self.cfg.static_inputs)].to_array().any("variable")
        
        # combine masks
        self.mask = self.target_mask | self.head_mask | self.static_mask

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()
            self.mask = self.mask.load()

        # Find indices of valid cells (not mask)
        if self.period != "test":
            self.coord_cells = np.argwhere(~self.mask.values) # indices of non-zero
        elif self.period == "test":
            # during test, don't need to mask so to reconstruct original dataset shape
            self.temp = np.ones(self.mask.shape).astype(bool)
            self.coord_cells = np.argwhere(self.temp)

        # Linear index, map integer sequence to tuple of (lat, lon) coordinates
        self.cell_linear_index = np.arange(0, len(self.coord_cells), 1)

        # Compute time index
        #if self.cfg.downsampling_temporal_dynamic or self.period == "test":
        self.time_index = np.arange(0, len(self.xd.time.values), 1)
        # else:
        #     self.time_index = np.arange((self.seq_len -1), # size to index 
        #                                 len(self.xd.time.values), # not inclusive 
        #                                 1)

        # Reduce dataset size
        if self.downsampler is not None:
            self.cell_linear_index, self.time_index = self.downsampler.sampling_idx([self.cell_linear_index, self.time_index])

        # Generate dataset indices 
        #if self.cfg.downsampling_temporal_dynamic:
            # This assumes that the time index for sampling the sequences are generated at runtime.
            # The dataset returns the whole time series for each data sample
            # In this way it is possible to generate new random time indices every epoch to dynamically subsample the time domain. 
        self.coord_samples = self.coord_cells[self.cell_linear_index]
        #else:
            # Combined cell and time indices
        #    self.coord_samples = list(itertools.product(*(self.coord_cells[self.cell_linear_index].tolist(), self.time_index.tolist() )))  

        # Normalize
        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("lat","lon", "time")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat","lon")
        )



        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")
        self.xs = self.scaler.transform(self.xs, "static_inputs")
        #
        if cfg.scaling_rescale_target is not None:
            # target has been transformed to vwc training statistics
            # now it needs to be scaled to minmax or whatever
            self.scaler.load_or_compute(
                self.y, 
                "target_variables", 
                is_train=True, # force comput stats
                axes=("lat", "lon", "time") # pixel by pixel
            )
            #self.y = self.scaler.transform(self.y, "target_variables")

        else:
            #pass
            self.scaler.load_or_compute(
                self.y, "target_variables", is_train, axes=("lat", "lon", "time")
            )
            #self.y = self.scaler.transform(self.y, "target_variables")
            

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")

    def rescale_target(self, urls, data_dynamic, data_target, xarray_kwargs = {}):
        if config := self.cfg.scaling_rescale_target.get("soil-property", False):
            lower = self.cfg.scaling_rescale_target["lower"]
            upper = self.cfg.scaling_rescale_target["upper"]
            par = read_from_zarr(url=urls["static_parameter_inputs"], chunks="auto", **xarray_kwargs)[[lower,upper]]
            self.y = self.rescale_target(self.y, par[lower], par[upper])
        elif config := self.cfg.scaling_rescale_target.get("model-statistics", False):
            # In inference calibration, rescale output surrogate to training target statistics
            # For the same period of training the surrogate
            # TODO: the period should be passed dynamically
            vs = data_dynamic[config.get("variable")].sel(time=slice("2017-01-01","2019-12-31"))
            if config.get("method") == "zscore":
                self.sim_std = vs.std("time")
                self.sim_mean = vs.mean("time")
                self.obs_std = self.y.ssm.std("time")
                self.obs_mean = self.y.ssm.mean("time")
                
                self.y = (self.sim_std / self.obs_std) * (self.y - self.obs_mean) + self.sim_mean

                self.obs_std = torch.from_numpy(self.obs_std.values).float()
                self.obs_mean = torch.from_numpy(self.obs_mean.values).float()
                self.sim_std = torch.from_numpy(self.sim_std.values).float()
                self.sim_mean = torch.from_numpy(self.sim_mean.values).float()
            elif config.get("method") == "minmax":
                self.sim_min = vs.min("time")
                self.sim_max = vs.max("time")
                self.obs_min = self.y.ssm.min("time")
                self.obs_max = self.y.ssm.max("time")

                self.y = (self.sim_max - self.sim_min) / (self.obs_max - self.obs_min) * (self.y - self.obs_min) + self.sim_min



    def __len__(self):
        return len((range(len(self.coord_samples))))

    def __getitem__(self, index):

        #if self.cfg.downsampling_temporal_dynamic or self.period != "test":

        idx_lat, idx_lon = self.coord_samples[index]

        ds_pixel_dynamic = self.xd.isel(lat=idx_lat, lon=idx_lon) # lat, lon, time -> time
        ds_pixel_target = self.y.isel(lat=idx_lat, lon=idx_lon)
        ds_pixel_static = self.xs.isel(lat=idx_lat, lon=idx_lon)

        ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
        ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature

        ds_pixel_static = ds_pixel_static.to_array()
        
        # TODO: remove call to float
        xd  = torch.tensor(ds_pixel_dynamic.values).float()
        xs = torch.tensor(ds_pixel_static.values).float()
        y = torch.tensor(ds_pixel_target.values).float()

        # else:
        #     idx_cell, idx_time = self.coord_samples[index]

        #     idx_lat, idx_lon = idx_cell
        #     # TODO: check
        #     ds_pixel_dynamic = self.xd.isel(
        #                                     lat=idx_lat, 
        #                                     lon=idx_lon, 
        #                                     time=slice(idx_time - (self.seq_len -1), # size seq_len to index 
        #                                                idx_time + 1) # not inclusive
        #                                                ) 

        #     ds_pixel_target = self.y.isel(  
        #                                     lat=idx_lat, 
        #                                     lon=idx_lon, 
        #                                     time=slice(idx_time - (self.seq_len - 1), idx_time + 1)) 
            
        #     ds_pixel_static = self.xs.isel(lat=idx_lat, lon=idx_lon)
    
        #     ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
        #     ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature
        #     ds_pixel_static = ds_pixel_static.to_array()
            
        #     # TODO: remove call to float
        #     xd  = torch.tensor(ds_pixel_dynamic.values).float()
        #     xs = torch.tensor(ds_pixel_static.values).float()
        #     y = torch.tensor(ds_pixel_target.values).float()

        return {"xd": xd, "xs": xs, "y": y}

class WflowSBMCube(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler
        self.cfg = cfg

        self.preprocessor = Preprocessor(cfg)

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        urls, xarray_kwargs = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")

        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto", **xarray_kwargs).sel(time=self.period_range)
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto",**xarray_kwargs)

        self.xd = data_dynamic[self.to_list(cfg.dynamic_inputs)]
        self.xs = data_static[self.to_list(cfg.static_inputs)]
        self.y = data_dynamic[self.to_list(cfg.target_variables)]

        
        self.shape = self.xd[self.cfg.dynamic_inputs[0]].shape

        if self.cfg.mask_variables is not None and self.period != "test":
            # apply mask 
            mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            self.mask = mask
        elif self.period == "test": # no masking when period is test 
            # FIXME mask is still useful
            mask = data_static[self.to_list(self.cfg.mask_variables)].to_array().any("variable")
            self.mask = mask


        (
            self.cbs_spatial_idxs,
            self.cbs_missing_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_spatial_slices,
        ) = compute_cubelet_spatial_idxs(
            self.shape,
            self.cfg.batch_size["xsize"],
            self.cfg.batch_size["ysize"],
            self.cfg.batch_overlap["xover"],
            self.cfg.batch_overlap["yover"],
            self.cfg.keep_spatial_degenerate_cubelet,
            masks=self.mask,
            missing_policy=self.cfg.missing_policy,
        )

        (
            self.cbs_time_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_time_slices,
        ) = compute_cubelet_time_idxs(
            self.shape,
            self.cfg.batch_size["tsize"],
            self.cfg.batch_overlap["tover"],
            self.cfg.keep_temporal_degenerate_cubelet,
            masks=self.mask,
        )

        cbs_tuple_idxs = compute_cubelet_tuple_idxs(
            self.cbs_spatial_idxs, self.cbs_time_idxs
        )
        cbs_slices = compute_cubelet_slices(
            self.cbs_spatial_slices, self.cbs_time_slices
        )

        self.cbs_mapping_idxs = cbs_mapping_idx_slice(cbs_tuple_idxs, cbs_slices)

        if self.downsampler is not None:
            # DOWNSAMPLE THE REMAINING INDEXES AFTER REMOVING MISSING AND DEGENERATED
            # return a subset of the cbs_mapping_idxs
            # TODO: also self.cbs_time_idxs and self.cbs_spatial_idxs should be updated
            self.cbs_mapping_idxs = self.downsampler.sampling_idx(self.cbs_mapping_idxs)

        if self.cfg.get("preprocessor") is not None:
            self.xs = self.preprocessor.process(self.xs, "static_inputs")
            self.xd = self.preprocessor.process(self.xd, "dynamic_inputs")
            self.y = self.preprocessor.process(self.y, "target_variables")

        # == SOME USEFUL PARAMETERS
        self.lat_size = len(self.xd.lat)
        self.lon_size = len(self.xd.lon)
        self.time_size = len(self.xd.time)
        self.dynamic_coords = self.xd.coords
        self.static_coords = self.xs.coords


        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("time", "lat", "lon")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat", "lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("time", "lat", "lon")
        )

        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")

        self.y = self.scaler.transform(self.y, "target_variables")

        self.xs = self.scaler.transform(self.xs, "static_inputs")

        if is_train: # write if train
            if not self.scaler.use_cached: # write if not reading from cache
                self.scaler.write("dynamic_inputs")
                self.scaler.write("static_inputs")
                self.scaler.write("target_variables")
            else: # if reading from cache
                if self.scaler.flag_stats_computed: # if stats were not found in cache
                    self.scaler.write("dynamic_inputs")
                    self.scaler.write("static_inputs")
                    self.scaler.write("target_variables")

        xd_data_vars = list(self.xd.data_vars)
        self.xd = self.xd.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        )  # time, lat, lon , feat
        self.xd = self.xd.transpose("time", "feat", "lat", "lon")  # T C H W
        self.xd = self.xd.astype("float32")
        self.xd = self.xd.drop_vars(["feat", "variable"]).assign_coords(
            {"feat": xd_data_vars}
        )

        y_data_vars = list(self.y.data_vars)
        self.y = self.y.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        )
        self.y = self.y.transpose("time", "feat", "lat", "lon")  # T C H W
        self.y = self.y.astype("float32")
        self.y = self.y.drop_vars(["feat", "variable"]).assign_coords(
            {"feat": y_data_vars}
        )

        xs_data_vars = list(self.xs.data_vars)
        self.xs = self.xs.to_stacked_array(
            new_dim="feat", sample_dims=["lat", "lon"]
        )  # H W C
        self.xs = self.xs.transpose("feat", "lat", "lon")
        self.xs = self.xs.astype("float32")
        self.xs = self.xs.drop_vars(["feat", "variable"]).assign_coords(
            {"feat": xs_data_vars}
        )

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()

        # TODO: improve this
        self.xd = self.xd.fillna(self.cfg.fill_missing)
        self.y = self.y.fillna(self.cfg.fill_missing)
        self.xs = self.xs.fillna(self.cfg.fill_missing)

    def __len__(self):
        return len(self.cbs_mapping_idxs)

    def get_indexes(self):
        return list(range(len(self.cbs_mapping_idxs)))

    def __getitem__(self, index):

        cubelet_idx = list(self.cbs_mapping_idxs.keys())[index]

        time_slice = self.cbs_mapping_idxs[cubelet_idx]["time"]
        lat_slice = self.cbs_mapping_idxs[cubelet_idx]["lat"]
        lon_slice = self.cbs_mapping_idxs[cubelet_idx]["lon"]

        # xr.Dataarray to np.ndarray, this triggers loading in memory, in case persist = False
        xd = self.xd[time_slice, :, lat_slice, lon_slice].values  # L C H W
        y = self.y[time_slice, :, lat_slice, lon_slice].values  # L C H W

        # np.ndarray ot torch.tensor
        xd = torch.tensor(xd)
        y = torch.tensor(y)

        xs = self.xs[:, lat_slice, lon_slice].values  # C H W
        xs = torch.tensor(xs)

        if self.cfg.static_to_dynamic:
            xs = xs.unsqueeze(0).repeat(xd.size(0), 1, 1, 1)

        return {"xd": xd, "xs": xs, "y": y}

class Wflow2dCal(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler
        self.cfg = cfg

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = slice(*cfg[f"{period}_temporal_range"])

        file_path = f"{cfg.data_dir}/{cfg.data_file}"

        self.xd = read_from_zarr(
            url="/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/adg1km_eobs_original.zarr",
            group="xd",
        ).sel(time=self.period)[list(self.cfg.dynamic_inputs)]

        self.xs = read_from_zarr(
            url="/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/param_learning_input/predictor_test.zarr"
        ).drop_vars("spatial_ref")[list(self.cfg.static_inputs)]

        self.y = xr.open_dataset(
            "/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed/daily/adige_2018-2021.nc",
            mask_and_scale=True,
        ).sel(time=self.period)

        self.shape = self.xd[self.cfg.dynamic_inputs[0]].shape

        self.xd = self.xd.rio.set_crs(4326).rio.write_crs()
        self.xd.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

        self.xd = self.xd.sel(time=self.y.time)

        self.xs = self.xs.rename({"x": "lon", "y": "lat"})

        if self.cfg.mask_variables is not None:
            self.mask = (
                read_from_zarr(url=file_path, group="mask")
                .mask.sel(mask_layer=self.cfg.mask_variables)
                .any(dim="mask_layer")
            )
        else:
            self.mask = None

        (
            self.cbs_spatial_idxs,
            self.cbs_missing_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_spatial_slices,
        ) = compute_cubelet_spatial_idxs(
            self.shape,
            self.cfg.batch_size["xsize"],
            self.cfg.batch_size["ysize"],
            self.cfg.batch_overlap["xover"],
            self.cfg.batch_overlap["yover"],
            self.cfg.keep_spatial_degenerate_cubelet,
            masks=self.mask,
            missing_policy=self.cfg.missing_policy,
        )

        (
            self.cbs_time_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_time_slices,
        ) = compute_cubelet_time_idxs(
            self.shape,
            self.cfg.batch_size["tsize"],
            self.cfg.batch_overlap["tover"],
            self.cfg.keep_temporal_degenerate_cubelet,
            masks=self.mask,
        )

        cbs_tuple_idxs = compute_cubelet_tuple_idxs(
            self.cbs_spatial_idxs, self.cbs_time_idxs
        )
        cbs_slices = compute_cubelet_slices(
            self.cbs_spatial_slices, self.cbs_time_slices
        )

        self.cbs_mapping_idxs = cbs_mapping_idx_slice(cbs_tuple_idxs, cbs_slices)

        if self.downsampler is not None:
            # DOWNSAMPLE THE REMAINING INDEXES AFTER REMOVING MISSING AND DEGENERATED
            # return a subset of the cbs_mapping_idxs
            # TODO: also self.cbs_time_idxs and self.cbs_spatial_idxs should be updated
            self.cbs_mapping_idxs = self.downsampler.sampling_idx(self.cbs_mapping_idxs)

            # Scaling

        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("time", "lat", "lon")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("lat", "lon")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("time", "lat", "lon")
        )

        if not self.scale_ontraining:
            self.xd = self.scaler.transform(self.xd, "dynamic_inputs")
            self.xs = self.scaler.transform(self.xs, "static_inputs")
            self.y = self.scaler.transform(self.y, "target_variables")

        if is_train:
            self.scaler.write("dynamic_inputs")
            self.scaler.write("static_inputs")
            self.scaler.write("target_variables")

        self.xd = self.xd.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        )  # time, lat, lon , feat
        self.xd = self.xd.transpose("time", "feat", "lat", "lon")  # T C H W
        self.xd = self.xd.astype("float32")

        self.y = self.y.to_stacked_array(
            new_dim="feat", sample_dims=["time", "lat", "lon"]
        )
        self.y = self.y.transpose("time", "feat", "lat", "lon")  # T C H W
        self.y = self.y.astype("float32")

        self.xs = self.xs.to_stacked_array(
            new_dim="feat", sample_dims=["lat", "lon"]
        )  # H W C
        self.xs = self.xs.transpose("feat", "lat", "lon")
        self.xs = self.xs.astype("float32")

        if self.cfg.persist:
            self.xd = self.xd.compute()
            self.y = self.y.compute()
            self.xs = self.xs.compute()

        # TODO: fix this
        self.xd = self.xd.fillna(self.cfg.fill_missing)
        self.y = self.y.fillna(self.cfg.fill_missing)
        self.xs = self.xs.fillna(self.cfg.fill_missing)

        self.top_layer_res = -1 * self.forcing.lat.diff("lat").values[0] / 2

    def __len__(self):
        return len(self.cbs_mapping_idxs)

    def get_indexes(self):
        return list(range(len(self.cbs_mapping_idxs)))

    def __getitem__(self, index):
        cubelet_idx = list(self.cbs_mapping_idxs.keys())[index]

        # print(index, cubelet_idx)

        time_slice = self.cbs_mapping_idxs[cubelet_idx]["time"]
        lat_slice = self.cbs_mapping_idxs[cubelet_idx]["lat"]
        lon_slice = self.cbs_mapping_idxs[cubelet_idx]["lon"]

        # xr.Dataarray to np.ndarray, this triggers loading in memory, in case persist = False
        forcing = self.xd[time_slice, :, lat_slice, lon_slice]  # .values # L C H W
        target = self.y[time_slice, :, lat_slice, lon_slice]  # .values # L C H W

        latmin = forcing.lat.values.min()
        latmax = forcing.lat.values.max()
        lonmin = forcing.lon.values.min()
        lonmax = forcing.lon.values.max()

        predictor = self.xs.sel(
            lat=slice(latmax + self.top_layer_res, latmin - self.top_layer_res),
            lon=slice(lonmin - self.top_layer_res, lonmax + self.top_layer_res),
        ).values

        forcing = forcing.values
        target = target.values

        # np.ndarray ot torch.tensor
        forcing = torch.FloatTensor(forcing)
        target = torch.FloatTensor(target)
        predictor = torch.FloatTensor(predictor)

        if self.cfg.lstm_1d:
            # Super slow when persist == False
            # If True means that the xsize and ysize is equal to 1

            # xd = xd.flatten(2,3) # L C H W => L C N
            xd = xd.squeeze()  # L C H W, but H W is size 1,1 => L C
            # xd = torch.permute(xd, (2, 0, 1)) # N L C, , but N = 1
            # xd = x.squeeze(0)

            # y = y.flatten(2,3) # L C H W => L C N
            y = y.squeeze()  # L C H W, but H W is size 1,1 => L C
            # y = torch.permute(y, (2, 0, 1)) # N L C, but N = 1
            # y = y.squeeze(0)
            if self.xs is not None:
                xs = xs.squeeze()  # C H W => C N

        if self.static_to_dynamic:
            if self.lstm_1d:
                # print(xs.shape)
                predictor = predictor.unsqueeze(0).repeat(
                    forcing.size(0),
                    1,
                )
            else:
                predictor = predictor.unsqueeze(0).repeat(forcing.size(0), 1, 1, 1)
        return predictor, forcing, target
