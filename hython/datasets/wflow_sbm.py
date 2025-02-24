"""Wflow_sbm emulators"""
from . import *
import itertools
    
class WflowSBM(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler
        self.cfg = cfg

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        urls = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")

        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto").sel(time=self.period_range).isel(lat=slice(None, None, -1))
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto")

        self.xd = data_dynamic[OmegaConf.to_object(cfg.dynamic_inputs)]
        self.xs = data_static[OmegaConf.to_object(cfg.static_inputs)]
        self.y = data_dynamic[OmegaConf.to_object(cfg.target_variables)]

        if not self.cfg.data_lazy_load: # loading in memory
            self.xd = self.xd.load()
            self.xs = self.xs.load()
            self.y = self.y.load()

        if self.cfg.mask_variables is not None and self.period != "test":
            # apply mask 
            mask = data_static[OmegaConf.to_object(self.cfg.mask_variables)].to_array().any("variable")
            self.mask = mask
            self.coords = np.argwhere(~mask.values)
        elif self.period == "test": # no masking when period is test 
            shape = list(self.xs.dims.values())
            self.coords =  np.argwhere(np.ones(shape).astype(bool))

        # compute cell index 
        self.cell_index = np.arange(0, len(self.coords), 1)
        
        # compute time index
        if self.cfg.downsampling_temporal_dynamic or self.period == "test":
            self.time_index = np.arange(0, len(self.xd.time.values), 1)
        else:
            self.time_index = np.arange(self.seq_len, len(self.xd.time.values), 1)
        
        # downsample indices based on rule
        if self.downsampler is not None:
            self.cell_index, self.time_index = self.downsampler.sampling_idx([self.cell_index, self.time_index])

        # generate dataset indices 
        if self.cfg.downsampling_temporal_dynamic:
            # This assumes that the time index for sampling the sequences are generated at runtime.
            # In this way it is possible to generate new random time indices every epoch to dynamically subsample the time domain. 
            self.coord_samples = self.coords[self.cell_index]
        else:
            # Combined cell and time indices
            self.coord_samples = list(itertools.product(*(self.coords[self.cell_index].tolist(), self.time_index.tolist() )))  

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
                scaling_static_reordered = {
                    k: self.cfg.scaling_static_range[k]
                    for k in self.cfg.static_inputs
                    if k in self.cfg.scaling_static_range
                }

                self.static_scale, self.static_center = self.get_scaling_parameter(
                    scaling_static_reordered, self.cfg.static_inputs
                )

                self.xs = self.scaler.transform_custom_range(
                    self.xs, "static_inputs", self.static_scale, self.static_center
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
    """ """

    def __init__(self, cfg, scaler, is_train=True, period="train"):
        super().__init__()

        self.cfg = cfg
        self.scaler = scaler

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        urls = get_source_url(cfg)

        # load datasets
        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto").sel(time=self.period_range).isel(lat=slice(None, None, -1))
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto")
        data_target = read_from_zarr(url=urls["target_variables"], chunks="auto").sel(time=self.period_range)
        
        # select 
        self.xs =  data_static[OmegaConf.to_object(cfg.static_inputs)]
        self.xd = data_dynamic[OmegaConf.to_object(cfg.dynamic_inputs)]
        self.y = data_target[OmegaConf.to_object(cfg.target_variables)]

        # TODO: ensure they are all float32

        # head_layer mask
        head_mask = read_from_zarr(url=urls["mask_variables"], chunks="auto")
        self.head_mask = head_mask[OmegaConf.to_object(self.cfg.mask_variables)].to_array().any("variable")
        
        # target mask, observation
        if urls.get("target_variables_mask", None):
            target_mask = read_from_zarr(url=urls["target_variables_mask"], chunks="auto").sel(time=self.period_range)
            sel_target_mask = OmegaConf.to_object(self.cfg.target_variables_mask)[0] if isinstance(OmegaConf.to_object(self.cfg.target_variables_mask), list) else OmegaConf.to_object(self.cfg.target_variables_mask)
            self.target_mask = target_mask[sel_target_mask]
            self.target_mask = self.target_mask.resample({"time":"1D"}).max().astype(bool)
            self.target_mask = self.target_mask.isnull().sum("time") > self.cfg.min_sample_target     
        else:
            self.target_mask = self.y.isnull().all("time")[OmegaConf.to_object(cfg.target_variables)[0]] #> self.cfg.min_sample_target   

        # static mask, predictors
        if urls.get("static_inputs_mask", None):
            self.static_mask = read_from_zarr(url=urls["static_inputs_mask"], chunks="auto")[OmegaConf.to_object(self.cfg.static_inputs_mask)[0]]
        else:
            self.static_mask = self.xs[OmegaConf.to_object(self.cfg.static_inputs)].to_array().any("variable")
        
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
            # in test, keep nans
            self.temp = np.ones(self.mask.shape).astype(bool)
            self.coord_cells = np.argwhere(self.temp)

        # Linear index, map to tuple of lat, lon coordinates
        self.cell_linear_index = np.arange(0, len(self.coord_cells), 1)

        # Compute time index
        if self.cfg.downsampling_temporal_dynamic or self.period == "test":
            self.time_index = np.arange(0, len(self.xd.time.values), 1)
        else:
            self.time_index = np.arange((self.seq_len -1), # size to index 
                                        len(self.xd.time.values), # not inclusive 
                                        1)

        # Reduce dataset size
        if self.downsampler is not None:
            self.cell_linear_index, self.time_index = self.downsampler.sampling_idx([self.cell_linear_index, self.time_index])

        # Generate dataset indices 
        if self.cfg.downsampling_temporal_dynamic:
            # This assumes that the time index for sampling the sequences are generated at runtime.
            # The dataset returns the whole time series for each data sample
            # In this way it is possible to generate new random time indices every epoch to dynamically subsample the time domain. 
            self.coord_samples = self.coord_cells[self.cell_linear_index]
        else:
            # Combined cell and time indices
            self.coord_samples = list(itertools.product(*(self.coord_cells[self.cell_linear_index].tolist(), self.time_index.tolist() )))  

        # Normalize
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
        self.xs = self.scaler.transform(self.xs, "static_inputs")
        self.y = self.scaler.transform(self.y, "target_variables")

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

        if self.cfg.downsampling_temporal_dynamic or self.period != "test":

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
        else:
            idx_cell, idx_time = self.coord_samples[index]

            idx_lat, idx_lon = idx_cell
            # TODO: check
            ds_pixel_dynamic = self.xd.isel(
                                            lat=idx_lat, 
                                            lon=idx_lon, 
                                            time=slice(idx_time - (self.seq_len -1), # size seq_len to index 
                                                       idx_time + 1) # not inclusive
                                                       ) 

            ds_pixel_target = self.y.isel(  
                                            lat=idx_lat, 
                                            lon=idx_lon, 
                                            time=slice(idx_time - (self.seq_len - 1), idx_time + 1)) 
            
            ds_pixel_static = self.xs.isel(lat=idx_lat, lon=idx_lon)
    
            ds_pixel_dynamic = ds_pixel_dynamic.to_array().transpose("time", "variable") # time -> time, feature
            ds_pixel_target = ds_pixel_target.to_array().transpose("time", "variable") # time -> time, feature
            ds_pixel_static = ds_pixel_static.to_array()
            
            # TODO: remove call to float
            xd  = torch.tensor(ds_pixel_dynamic.values).float()
            xs = torch.tensor(ds_pixel_static.values).float()
            y = torch.tensor(ds_pixel_target.values).float()

        return {"xd": xd, "xs": xs, "y": y}

class WflowSBMCube(BaseDataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler
        self.cfg = cfg

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        urls = get_source_url(cfg)

        self.scaling_static_range = self.cfg.get("scaling_static_range")

        data_dynamic = read_from_zarr(url=urls["dynamic_inputs"], chunks="auto").sel(time=self.period_range).isel(lat=slice(None, None, -1))
        data_static = read_from_zarr(url=urls["static_inputs"], chunks="auto")

        self.xd = data_dynamic[OmegaConf.to_object(cfg.dynamic_inputs)]
        self.xs = data_static[OmegaConf.to_object(cfg.static_inputs)]
        self.y = data_dynamic[OmegaConf.to_object(cfg.target_variables)]

        
        self.shape = self.xd[self.cfg.dynamic_inputs[0]].shape

        if self.cfg.mask_variables is not None and self.period != "test":
            # apply mask 
            mask = data_static[OmegaConf.to_object(self.cfg.mask_variables)].to_array().any("variable")
            self.mask = mask
            #self.coords = np.argwhere(~mask.values)
        elif self.period == "test": # no masking when period is test 
            shape = list(self.xs.dims.values())
            self.coords =  np.argwhere(np.ones(shape).astype(bool))


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

            self.y = self.scaler.transform(self.y, "target_variables")

            if self.scaling_static_range is not None:
                scaling_static_reordered = {
                    k: self.cfg.scaling_static_range[k]
                    for k in self.cfg.static_inputs
                    if k in self.cfg.scaling_static_range
                }

                self.static_scale, self.static_center = self.get_scaling_parameter(
                    scaling_static_reordered, self.cfg.static_inputs
                )

                self.xs = self.scaler.transform_custom_range(
                    self.xs, "static_inputs", self.static_scale, self.static_center
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

        #import pdb;pdb.set_trace()
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

        # TODO: fix this
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
