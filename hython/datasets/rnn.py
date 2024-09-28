
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
    ):
        self.shape = original_domain_shape

        self.xd = xd
        self.y = y
        self.xs = xs

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

            if normalizer_dynamic.stats_iscomputed:  # validation or test
                self.xd = normalizer_dynamic.normalize(self.xd)
            else:
                # compute stats for training
                normalizer_dynamic.compute_stats(self.xd[self.grid_idx_1d_valid])
                self.xd = normalizer_dynamic.normalize(self.xd)

        if normalizer_static is not None:
            if normalizer_static.stats_iscomputed:  # validation or test
                self.xs = normalizer_static.normalize(self.xs)
            else:
                if downsampler:
                    normalizer_static.compute_stats(self.xs[self.grid_idx_1d_valid])
                else:
                    normalizer_static.compute_stats(self.xs)

                self.xs = normalizer_static.normalize(self.xs)
        if normalizer_target is not None:
            if normalizer_target.stats_iscomputed:  # validation or test
                self.y = normalizer_target.normalize(self.y)
            else:
                normalizer_target.compute_stats(self.y[self.grid_idx_1d_valid])
                self.y = normalizer_target.normalize(self.y)

        self.xs = self.xs.astype(np.float32)

        if isinstance(self.xd, xr.DataArray):
            self.xd = torch.tensor(self.xd.values)
            self.y = torch.tensor(self.y.values)
            self.xs = torch.tensor(self.xs.values)
        else:
            self.xd = torch.tensor(self.xd)
            self.y = torch.tensor(self.y)
            self.xs = torch.tensor(self.xs)

    def __len__(self):
        return len(self.grid_idx_1d_valid)

    def get_indexes(self):
        return list(range(len(self.grid_idx_1d_valid)))

    def __getitem__(self, index):
        item_index = self.grid_idx_1d_valid[index]

        if self.xs is not None:
            return self.xd[item_index], self.xs[item_index], self.y[item_index]
        else:
            return self.xd[item_index], self.y[item_index]


