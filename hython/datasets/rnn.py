from . import *


class LSTMDataset(Dataset):
    def __init__(
        self,
        xd: xr.DataArray | npt.NDArray,
        y: xr.DataArray | npt.NDArray,
        xs: xr.DataArray | npt.NDArray = None,
        original_domain_shape=(),
        mask=None,
        downsampler=None,
        normalizer_dynamic=None,
        normalizer_static=None,
        normalizer_target=None,
        persist=False,
    ):
        self.shape = original_domain_shape

        self.persist = persist
        self.xd = xd
        self.y = y
        self.xs = xs

        self.xs = self.xs.astype(np.float32)

        self.downsampler = downsampler

        # compute indexes

        ishape = self.shape[0]  # rows (y, lat)
        jshape = self.shape[1]  # columns (x, lon)

        irange = np.arange(0, ishape, 1)
        jrange = np.arange(0, jshape, 1)

        self.grid_idx_2d = compute_grid_indices(shape=self.shape)
        self.grid_idx_1d = self.grid_idx_2d.flatten()

        # IF DOWNSAMPLING
        # Reduces the available indexes to a valid subset
        if downsampler:
            # Same keep only indexes that satisfy some rule
            self.grid_idx_1d_downsampled = self.downsampler.sampling_idx(
                self.grid_idx_2d, self.shape
            )

        # IF REMOVE MISSING FROM MASK
        # Reduces the available indexes to a valid subset
        if mask is not None:
            # This actually does not touch the dataset, only remove indexes corresponding to missing values from the available indexes
            idx_nan = self.grid_idx_2d[mask]

            if downsampler:
                self.grid_idx_1d_valid = np.setdiff1d(
                    self.grid_idx_1d_downsampled, idx_nan
                )
            else:
                self.grid_idx_1d_valid = np.setdiff1d(self.grid_idx_1d, idx_nan)
        else:
            if downsampler:
                self.grid_idx_1d_valid = self.grid_idx_1d_downsampled
            else:
                self.grid_idx_1d_valid = self.grid_idx_1d

        # NORMALIZE BASED IF MAKS AND IF DOWNSAMPLING
        if normalizer_dynamic is not None:
            # this normalize the data corresponding to valid indexes
            if not normalizer_dynamic.stats_iscomputed:  # compute only when train_data initialized
                normalizer_dynamic.compute_stats(self.xd[self.grid_idx_1d_valid])

        if normalizer_static is not None:
            if not normalizer_static.stats_iscomputed:  # compute only when train_data initialized
                normalizer_static.compute_stats(self.xs[self.grid_idx_1d_valid])

        if normalizer_target is not None:
            if not normalizer_target.stats_iscomputed:  # compute only when train_data initialized
                normalizer_target.compute_stats(self.y[self.grid_idx_1d_valid])

        self.n_dynamic = normalizer_dynamic
        self.n_static = normalizer_static
        self.n_target = normalizer_target

        if self.persist:
            self.xd = self.xd.persist().values
            self.y = self.y.persist().values
            self.xs = self.xs.persist().values

        

    def __len__(self):
        return len(self.grid_idx_1d_valid)

    def get_indexes(self):
        return list(range(len(self.grid_idx_1d_valid)))

    def __getitem__(self, index):
        item_index = self.grid_idx_1d_valid[index]
        
        if self.persist:
            xd = self.xd[item_index]
            xs = self.xs[item_index]
            y = self.y[item_index]
        else:
            xd = self.xd[item_index].values
            xs = self.xs[item_index].values
            y = self.y[item_index].values     

        # Normalize
        #import pdb;pdb.set_trace()
        if self.n_dynamic is not None:
            xd = self.n_dynamic.normalize(xd)[0]
        if self.n_static is not None:
            xs = self.n_static.normalize(xs)[0]
        if self.n_target is not None:
            y = self.n_target.normalize(y)[0]

        # to tensor 
        xd = torch.tensor(xd)
        y = torch.tensor(y)
        xs = torch.tensor(xs)

        if self.xs is not None:
            return  xd, xs, y
        else:
            return xd, xs
