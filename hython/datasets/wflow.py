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