"""Wflow_sbm emulators"""
from . import *


class Wflow1d(Dataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler
        self.cfg = cfg

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = slice(*cfg[f"{period}_temporal_range"])

        file_path = f"{cfg.data_dir}/{cfg.data_file}"

        # generate run directory
        run_path = generate_run_folder(cfg)

        self.xd = (
            read_from_zarr(url=file_path, group="xd", multi_index="gridcell")
            .sel(time=self.period)
            .xd.sel(feat=cfg.dynamic_inputs)
        )

        self.xs = read_from_zarr(
            url=file_path, group="xs", multi_index="gridcell"
        ).xs.sel(feat=cfg.static_inputs)

        self.y = (
            read_from_zarr(url=file_path, group="y", multi_index="gridcell")
            .sel(time=self.period)
            .y.sel(feat=cfg.target_variables)
        )

        self.shape = self.xd.attrs["shape"]

        # compute indexes

        self.grid_idx_2d = compute_grid_indices(shape=self.shape)
        self.grid_idx_1d = self.grid_idx_2d.flatten()

        if self.downsampler is not None:
            # Same keep only indexes that satisfy some rule
            self.grid_idx_1d_downsampled = self.downsampler.sampling_idx(
                self.grid_idx_2d, self.shape
            )

        if self.cfg.mask_variables is not None:
            masks = (
                read_from_zarr(url=file_path, group="mask")
                .mask.sel(mask_layer=self.cfg.mask_variables)
                .any(dim="mask_layer")
            )
            idx_nan = self.grid_idx_2d[masks]

            if self.downsampler is not None:
                self.grid_idx_1d_valid = np.setdiff1d(
                    self.grid_idx_1d_downsampled, idx_nan
                )
            else:
                self.grid_idx_1d_valid = np.setdiff1d(self.grid_idx_1d, idx_nan)
        else:
            if self.downsampler is not None:
                self.grid_idx_1d_valid = self.grid_idx_1d_downsampled
            else:
                self.grid_idx_1d_valid = self.grid_idx_1d

        # Scaling

        self.scaler.set_run_dir(run_path)

        self.scaler.load_or_compute(
            self.xd, "dynamic_inputs", is_train, axes=("gridcell", "time")
        )

        self.scaler.load_or_compute(
            self.xs, "static_inputs", is_train, axes=("gridcell")
        )

        self.scaler.load_or_compute(
            self.y, "target_variables", is_train, axes=("gridcell", "time")
        )

        if not self.scale_ontraining:
            self.xd = self.scaler.transform(self.xd, "dynamic_inputs")
            self.xs = self.scaler.transform(self.xs, "static_inputs")
            self.y = self.scaler.transform(self.y, "target_variables")

        if is_train:
            self.scaler.write("dynamic_inputs")
            self.scaler.write("static_inputs")
            self.scaler.write("target_variables")

    def __len__(self):
        return len(self.grid_idx_1d_valid)

    def __getitem__(self, index):
        item_index = self.grid_idx_1d_valid[index]

        if self.scale_ontraining:
            xd = torch.tensor(
                self.scaler.transform(self.xd[item_index], "dynamic_inputs").values
            ).float()
            xs = torch.tensor(
                self.scaler.transform(self.xs[item_index], "static_inputs").values
            ).float()
            y = torch.tensor(
                self.scaler.transform(self.y[item_index], "target_variables").values
            ).float()
        else:
            xd = torch.tensor(self.xd[item_index].values).float()
            xs = torch.tensor(self.xs[item_index].values).float()
            y = torch.tensor(self.y[item_index].values).float()

        return {"xd": xd, "xs": xs, "y": y}


class Wflow1dCal(Dataset):
    """
    """
    def __init__(self, cfg, scaler, is_train=True, period="train"):
        super().__init__()

        self.cfg = cfg
        self.scaler = scaler

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = period
        self.period_range = slice(*cfg[f"{period}_temporal_range"])

        file_path = f"{cfg.data_dir}/{cfg.data_file}"

        # generate run directory
        run_path = generate_run_folder(cfg)

        # load datasets
        self.static = (
            read_from_zarr(
                cfg.data_static_inputs,
                group="attr",
                multi_index="gridcell",
            )
            .sel(feat=self.cfg.static_inputs)
            .astype(np.float32)["attr"]
        )

        self.dynamic = (
            read_from_zarr(
                cfg.data_dynamic_inputs,
                group="xd",
                multi_index="gridcell",
            )
            .xd.sel(time=self.period_range)
            .sel(feat=self.cfg.dynamic_inputs)
        )
        

        self.obs = xr.open_dataset(
            cfg.data_target_variables,
            mask_and_scale=True,
        ).sel(time=self.period_range)

        # mask wflow
        self.wflow_mask = (
            read_from_zarr(url=file_path, group="mask")
            .mask.sel(mask_layer=self.cfg.mask_variables)
            .any(dim="mask_layer")
        )

        # mask observation
        obs_mask = xr.open_dataset(
            cfg.data_target_mask
        )
        obs_mask = obs_mask.resample({"time": "1D"}).max()
        obs_mask = obs_mask.astype(bool)

        self.obs = self.obs.where(obs_mask)

        self.obs_mask = (~self.obs.isnull()).sum("time").ssm > self.cfg.min_sample_target

        # mask predictors
        self.predictor_mask = (
            ~self.static.unstack().sortby(["lat", "lon"]).isnull().any("feat")
        )
        # combine masks
        self.mask = self.obs_mask & (~self.wflow_mask) & self.predictor_mask

        # (1) apply 2d mask
        obs_masked = self.obs.where(self.mask)

        # (2) reshape
        self.obs = reshape(self.obs)
        obs_masked = reshape(obs_masked)

        # (3) find indices of valid
        if self.period != "test":
            valid_coords = np.argwhere(~(obs_masked.isnull()).values.squeeze(-1))

            # (4) avoid hitting bounds
            self.coords = valid_coords[valid_coords[:, 1] > self.cfg.seq_length]
        else:
            self.temp = np.ones(obs_masked.shape).astype(bool)
            valid_coords = np.argwhere(self.temp.squeeze(-1))
           
            # keep only 
            _, index = np.unique(valid_coords[:, 0], return_index=True)

            self.coords = valid_coords[index]



        # (5) reduce dataset size
        if self.downsampler is not None:
            self.coords = self.downsampler.sampling_idx(self.coords)

        gridcell_idx = np.unique(self.coords[:, 0])
        time_idx = np.unique(self.coords[:, 1])

        # (6) Normalize

        self.scaler.set_run_dir(run_path)

        self.scaler.load_or_compute(
            self.dynamic.isel(gridcell=gridcell_idx, time=time_idx),
            "dynamic_inputs",
            is_train,
            axes=("gridcell", "time"),
        )
        self.dynamic = self.scaler.transform(self.dynamic, "dynamic_inputs")

        self.scaler.load_or_compute(
            self.static.isel(gridcell=gridcell_idx),
            "static_inputs",
            is_train,
            axes=("gridcell"),
        )
        self.static = self.scaler.transform(self.static, "static_inputs")

        self.scaler.load_or_compute(
            self.obs.isel(gridcell=gridcell_idx, time=time_idx),
            "target_variables",
            is_train,
            axes=("gridcell", "time"),
        )
        self.obs = self.scaler.transform(self.obs, "target_variables")

        self.obs = self.obs.astype(np.float32)
        self.static = self.static.astype(np.float32)
        self.dynamic = self.dynamic.astype(np.float32)

        if is_train:
            self.scaler.write("dynamic_inputs")
            self.scaler.write("static_inputs")
            self.scaler.write("target_variables")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        gridcell, time = self.coords[i]
        
        if self.period != "test":
            time_delta = slice(time - self.cfg.seq_length + 1, time + 1)

            xs = self.static.isel(gridcell=gridcell).values
            xd = self.dynamic.isel(gridcell=gridcell, time=time_delta).values
            yo = self.obs.isel(gridcell=gridcell, time=time_delta).values
        else:
            xs = self.static.isel(gridcell=gridcell).values
            xd = self.dynamic.isel(gridcell=gridcell).values
            yo = self.obs.isel(gridcell=gridcell).values

        return {"xd": xd, "xs": xs, "y": yo}


class Wflow2d(Dataset):
    def __init__(
        self, cfg, scaler, is_train=True, period="train", scale_ontraining=False
    ):
        self.scale_ontraining = scale_ontraining
        self.scaler = scaler
        self.cfg = cfg

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = slice(*cfg[f"{period}_temporal_range"])

        file_path = f"{cfg.data_dir}/{cfg.data_file}"

        self.xd = read_from_zarr(url=file_path, group="xd").sel(time=self.period)[
            list(self.cfg.dynamic_inputs)
        ]

        self.xs = read_from_zarr(url=file_path, group="xs")[
            list(self.cfg.static_inputs)
        ]

        self.y = read_from_zarr(url=file_path, group="y").sel(time=self.period)[
            list(self.cfg.target_variables)
        ]

        self.shape = self.xd[self.cfg.dynamic_inputs[0]].shape

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

        if self.cfg.persist:
            self.xd = self.xd.compute()
            self.y = self.y.compute()
            self.xs = self.xs.compute()

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

        if self.xs is not None:
            xs = self.xs[:, lat_slice, lon_slice].values  # C H W
            xs = torch.tensor(xs)
        else:
            xs = None

        if self.lstm_1d:
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

        if self.xs is not None:
            if self.static_to_dynamic:
                if self.lstm_1d:
                    xs = xs.unsqueeze(0).repeat(
                        xd.size(0),
                        1,
                    )
                else:
                    xs = xs.unsqueeze(0).repeat(xd.size(0), 1, 1, 1)
            return xd, xs, y
        else:
            return xd, torch.tensor([]), y


class Wflow2dCal(Dataset):
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
