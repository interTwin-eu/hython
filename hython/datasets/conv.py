from . import *


class CubeletsDataset(Dataset):
    def __init__(
        self,
        xd: xr.Dataset,  # time, lat, lon
        y: xr.Dataset,  # time, lat, lon
        xs: xr.Dataset = None,  # lat, lon
        mask=None,
        downsampler=None,
        normalizer_dynamic=None,
        normalizer_static=None,
        normalizer_target=None,
        shape: tuple = (),  # time ,lat ,lon
        batch_size: dict = {"xsize": 20, "ysize": 20, "tsize": 20},
        overlap: dict = {"xover": 0, "yover": 0, "tover": 0},
        missing_policy: str | float = "all",
        fill_missing=0,
        persist=False,
        lstm_1d=False,
        static_to_dynamic=False,
    ):
        self.xd = xd
        self.y = y
        self.xs = xs

        self.shape = shape

        self.mask = mask

        self.missing_policy = missing_policy

        self.downsampler = downsampler

        KEEP_DEGENERATE_CUBELETS = False  # TODO: hardcoded

        # compute stuff

        (
            self.cbs_spatial_idxs,
            self.cbs_missing_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_spatial_slices,
        ) = compute_cubelet_spatial_idxs(
            shape,
            batch_size["xsize"],
            batch_size["ysize"],
            overlap["xover"],
            overlap["yover"],
            KEEP_DEGENERATE_CUBELETS,
            masks=self.mask,
            missing_policy=self.missing_policy,
        )
        # print(self.cbs_spatial_idxs)

        (
            self.cbs_time_idxs,
            self.cbs_degenerate_idxs,
            self.cbs_time_slices,
        ) = compute_cubelet_time_idxs(
            shape,
            batch_size["tsize"],
            overlap["tover"],
            KEEP_DEGENERATE_CUBELETS,
            masks=self.mask,
        )

        # print(self.cbs_spatial_idxs)

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

        if normalizer_dynamic is not None:
            # this normalize the data corresponding to valid indexes
            if normalizer_dynamic.stats_iscomputed:  # validation or test
                self.xd = normalizer_dynamic.normalize(self.xd)
            else:
                # compute stats for training
                normalizer_dynamic.compute_stats(self.xd)
                self.xd = normalizer_dynamic.normalize(self.xd)

        if normalizer_static is not None:
            # import pdb;pdb.set_trace()
            if normalizer_static.stats_iscomputed:  # validation or test
                self.xs = normalizer_static.normalize(self.xs)
            else:
                normalizer_static.compute_stats(self.xs)
                self.xs = normalizer_static.normalize(self.xs)

        if normalizer_target is not None:
            if normalizer_target.stats_iscomputed:  # validation or test
                self.y = normalizer_target.normalize(self.y)
            else:
                normalizer_target.compute_stats(self.y)
                self.y = normalizer_target.normalize(self.y)

        # either do this here or in the getitem
        # maybe add HERE THE RASHAPE IN CASE IS LSTM 1D
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

        if self.xs is not None:
            xs_data_vars = list(self.xs.data_vars)
            self.xs = self.xs.to_stacked_array(
                new_dim="feat", sample_dims=["lat", "lon"]
            )  # H W C
            self.xs = self.xs.transpose("feat", "lat", "lon")
            self.xs = self.xs.astype("float32")
            self.xs = self.xs.drop_vars(["feat", "variable"]).assign_coords(
                {"feat": xs_data_vars}
            )

        if persist:
            self.xd = self.xd.compute()
            self.y = self.y.compute()
            if self.xs is not None:
                self.xs = self.xs.compute()

        # The conversion of missing nans should occur at the very end. The nans are used in the compute_cubelets_*_idxs to remove cubelets with all nans
        # should the missing flag not be equal to a potential valid value for that quantity? for example zero may be valid for many geophysical variables
        self.xd = self.xd.fillna(fill_missing)
        self.y = self.y.fillna(fill_missing)

        if self.xs is not None:
            self.xs = self.xs.fillna(fill_missing)

        self.lstm_1d = lstm_1d
        self.static_to_dynamic = static_to_dynamic

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


class XBatchDataset(Dataset):
    """
    Returns batches of Ntile,seq L T C H W

    The problem with this shape is that I cannot sample in time as the data points mixes both space (tile) and time (seq)

    """

    def __init__(
        self, xd, y, xs=None, lstm=False, join=False, xbatcher_kwargs={}, transform=None
    ):
        self.join = join

        self.xd_gen = self._get_xbatcher_generator(xd, **xbatcher_kwargs)

        print("dynamic: ", len(self.xd_gen))

        self.y_gen = self._get_xbatcher_generator(y, **xbatcher_kwargs)

        # xbatcher_kwargs["input_dims"].pop("time")

        self.xs_gen = self._get_xbatcher_generator(xs, **xbatcher_kwargs)
        self.transform = transform
        self.lstm = lstm

        print("static: ", len(self.xs_gen))

    def __len__(self):
        return len(self.xd_gen)

    def __getitem__(self, index):
        if self.lstm:
            # Now using this: Ntile[index] C Nseq L H W => Npixel L C
            ## Ntile,seq[index] C L H W => Npixel L C

            xd = self.xd_gen[index].torch.to_tensor().flatten(2)  # C L Npixel
            # print(xd.shape)
            xd = torch.permute(xd, (2, 1, 0))  # Npixel L C
            # print(xd.shape)
            y = self.y_gen[index].torch.to_tensor().flatten(2)  # C L Npixel
            y = torch.permute(y, (2, 1, 0))  # # Npixel L C

            if self.xs_gen is not None:
                # Ntile[index] C H W
                # Ntile,seq[index] C L H W
                xs = self.xs_gen[index].torch.to_tensor().flatten(2)  # C L Npixel
                # print(xs.shape)
                # xs = xs.unsqueeze(1).repeat(1, xd.size(1), 1) # C T Npixel
                # xs = xs.unsqueeze(2).repeat(1, 1, xd.size(2), 1) # C T Nseq Npixel

        else:
            # Ntile,seq[index] C T H W => Npixel T C

            xd = (
                self.xd_gen[index].transpose("time", "variable", ...).torch.to_tensor()
            )  # C T H W => T C H W
            y = (
                self.y_gen[index].transpose("time", "variable", ...).torch.to_tensor()
            )  # C T H W => T C H W

            if self.xs_gen is not None:
                xs = (
                    self.xs_gen[index]
                    .transpose("time", "variable", ...)
                    .torch.to_tensor()
                )
                xs = xs.to(torch.float32)  # torch.transpose(xs, 0,1)
                # import pdb;pdb.set_trace()
                # xs = xs.unsqueeze(0).repeat(xd.size(0), 1, 1, 1) # T C H W

        if self.transform:
            xd = self.transform(xd)

        if self.xs_gen is not None:
            if self.join:
                return xd, y
            else:
                return xd, xs, y
        else:
            return xd, y

    def _get_xbatcher_generator(
        self,
        ds,
        input_dims,
        concat_input_dims=True,
        batch_dims={},
        input_overlap={},
        preload_batch=False,
    ):
        time = input_dims.get("time", None)

        if ds is None:
            return None

        if time is None:
            gen = xbatcher.BatchGenerator(
                ds=ds,
                input_dims=input_dims,
                concat_input_dims=False,
                batch_dims=batch_dims,
                input_overlap=input_overlap,
                preload_batch=preload_batch,
            )
        else:
            gen = xbatcher.BatchGenerator(
                ds=ds,
                input_dims=input_dims,
                concat_input_dims=concat_input_dims,
                batch_dims=batch_dims,
                input_overlap=input_overlap,
                preload_batch=preload_batch,
            )

        return gen