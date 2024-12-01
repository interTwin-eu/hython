from . import *


class Wflow1d(Dataset):
    def __init__(
        self,
        cfg,
        scaler,
        is_train=True,
        period="train"
    ):
        
        self.scaler = scaler 
        self.cfg = cfg 

        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = slice(*cfg[f"{period}_temporal_range"])

        file_path = f"{cfg.data_dir}/{cfg.data_file}"

        self.xd = (
            read_from_zarr(url=file_path, group="xd", multi_index="gridcell")
            .sel(time=self.period)
            .xd.sel(feat=cfg.dynamic_inputs)
        )
        self.xs = read_from_zarr(url=file_path, group="xs", multi_index="gridcell").xs.sel(
            feat=cfg.static_inputs
        )
        self.y = (
            read_from_zarr(url=file_path, group="y", multi_index="gridcell")
            .sel(time=self.period)
            .y.sel(feat=cfg.target_variables)
        )

        self.shape = self.xd.attrs["shape"]

        # compute indexes

        ishape = self.shape[0]  # rows (y, lat)
        jshape = self.shape[1]  # columns (x, lon)

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

        self.scaler.load_or_compute(self.xd, "dynamic_inputs", is_train, axes=("gridcell","time"))
        self.xd = self.scaler.transform(self.xd, "dynamic_inputs")


        self.scaler.load_or_compute(self.xs, "static_inputs", is_train, axes=("gridcell"))
        self.xs = self.scaler.transform(self.xs, "static_inputs")


        self.scaler.load_or_compute(self.y, "target_variables", is_train, axes=("gridcell","time"))
        self.y = self.scaler.transform(self.y, "target_variables")

        if is_train:
            self.scaler.write("dynamic_inputs")
            self.scaler.write("static_inputs")
            self.scaler.write("target_variables")

        

    def __len__(self):
        return len(self.grid_idx_1d_valid)
        

    def __getitem__(self, index):
        item_index = self.grid_idx_1d_valid[index]

        xd = torch.tensor(self.xd[item_index].values).float()
        xs = torch.tensor(self.xs[item_index].values).float()
        y = torch.tensor(self.y[item_index].values).float()

        return {"xd":xd, 
                "xs":xs, 
                "y":y
                }
        


class Wflow2d(Dataset):
    pass


class Wflow1dCal(Dataset):
    def __init__(self, 
                 cfg, 
                 scaler,
                 is_train= True,  
                 period="train"
                 ):
        
        super().__init__()
        
        self.cfg = cfg
        self.scaler = scaler
        
        self.downsampler = self.cfg[f"{period}_downsampler"]

        self.period = slice(*cfg[f"{period}_temporal_range"])

        file_path = f"{cfg.data_dir}/{cfg.data_file}"

        # load datasets
        self.static = read_from_zarr("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/param_learning_input/predictor_lstm.zarr",
                                    group="attr",
                                    multi_index="gridcell").sel(feat=self.cfg.static_inputs).astype(np.float32)["attr"]
        
        self.dynamic = (read_from_zarr("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/surrogate_input/adg1km_eobs_preprocessed.zarr", 
                                       group="xd", multi_index="gridcell")
                            .xd.sel(time=self.period)
                            .sel(feat=self.cfg.dynamic_inputs) 
                        )
        
        self.obs = (xr.open_dataset("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed/daily/adige_2018-2021.nc", 
                                    mask_and_scale=True)
                            .sel(time=self.period)
                            )

        # mask wflow
        wflow_mask = (
                read_from_zarr(url=file_path, group="mask")
                .mask.sel(mask_layer=self.cfg.mask_variables)
                .any(dim="mask_layer"))

        # mask observation
        obs_mask = xr.open_dataset("/mnt/CEPH_PROJECTS/InterTwin/hydrologic_data/SSM-RT0-SIG0-R-CRRL/processed/adige_mask_2017_2022.nc")
        obs_mask = obs_mask.resample({"time":"1D"}).max()
        obs_mask = obs_mask.astype(bool)

        self.obs = self.obs.where(obs_mask)

        obs_mask = (~self.obs.isnull()).sum("time").ssm > self.cfg.min_sample_target

        # mask predictors
        predictor_mask = (~self.static.unstack().sortby(["lat","lon"]).isnull().any("feat"))
        # combine masks
        self.mask = obs_mask & (~wflow_mask) & predictor_mask

    
        # (1) apply 2d mask
        obs_masked = self.obs.where(self.mask)
        
        # (2) reshape
        self.obs = reshape(self.obs)
        obs_masked = reshape(obs_masked)
        
        # (3) find indices of valid
        valid_coords = np.argwhere(~(obs_masked.isnull()).values.squeeze(-1))
        
        # (4) avoid hitting bounds
        self.coords = valid_coords[valid_coords[:,1] > self.cfg.seq_length]

        # (5) reduce dataset size
        if self.downsampler is not None:
            self.coords = self.downsampler.sampling_idx(self.coords)

        gridcell_idx = np.unique(self.coords[:,0])
        time_idx = np.unique(self.coords[:,1])
        
        # (6) Normalize
        
        self.scaler.load_or_compute(self.dynamic.isel(gridcell=gridcell_idx, time=time_idx), 
                                    "dynamic_inputs", is_train, axes=("gridcell","time"))
        self.dynamic = self.scaler.transform(self.dynamic, "dynamic_inputs")


        self.scaler.load_or_compute(self.static.isel(gridcell=gridcell_idx), 
                                    "static_inputs", is_train, axes=("gridcell"))
        self.static = self.scaler.transform(self.static, "static_inputs")


        self.scaler.load_or_compute(self.obs.isel(gridcell=gridcell_idx, time=time_idx),
                                     "target_variables", is_train, axes=("gridcell","time"))
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

        time_delta = slice(time - self.cfg.seq_length + 1, time + 1)
        
        xs = self.static.isel(gridcell=gridcell).values
        xd = self.dynamic.isel(gridcell=gridcell, time=time_delta).values
        yo = self.obs.isel(gridcell=gridcell, time=time_delta).values


        return {"xd":xd, "xs":xs, "y":yo}